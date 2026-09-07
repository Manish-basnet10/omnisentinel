"""
Feature Service — PCAP → CICFlowMeter-Compatible 60 Features
=============================================================
Uses scapy to extract per-packet statistics from a PCAP, computes
bidirectional flow statistics matching the exact 60-feature schema
that the trained OmniSentinel models expect.

Feature derivation follows the CICFlowMeter v3 specification:
  https://www.unb.ca/cic/research/applications.html#CICFlowMeter

Key guarantees:
  - Output always has EXACTLY len(feature_cols) columns.
  - Columns are in the exact order stored in feature_list.json.
  - No column is fabricated without disclosure.
  - The existing StandardScaler is NEVER re-fitted here.
"""
from __future__ import annotations

import math
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("omnisentinel.features")

# ── Scapy — lazy import (large library, slow startup) ────────────────────────
try:
    from scapy.utils import PcapReader
    from scapy.layers.inet import IP, TCP, UDP
    _SCAPY_OK = True
except ImportError:
    _SCAPY_OK = False
    logger.warning("[FEATURES] scapy not installed. Install with: pip install scapy")

# ── CICFlowMeter constants ────────────────────────────────────────────────────
ACTIVE_TIMEOUT = 5_000_000    # 5 seconds in microseconds (CICFlowMeter default)
MAX_FLOWS      = 10_000       # cap to prevent OOM on huge PCAPs


# ══════════════════════════════════════════════════════════════════════════════
# Flow Record
# ══════════════════════════════════════════════════════════════════════════════
class _FlowRecord:
    """
    Accumulates per-packet statistics for one bidirectional flow.
    The 'forward' direction is always the direction of the FIRST packet seen.
    """
    __slots__ = (
        "key", "dst_port", "first_ts", "last_ts",
        "fwd_lengths", "bwd_lengths",
        "all_ts", "fwd_ts", "bwd_ts",
        "fin", "rst", "psh", "ack", "urg",
        "fwd_psh", "fwd_urg",
        "init_win_fwd", "init_win_bwd",
        "fwd_seg_sizes",
        "active_periods", "last_active_start", "last_active_end",
        "idle_gaps",
    )

    def __init__(
        self,
        key: tuple,
        ts: float,
        pkt_len: int,
        tcp_flags: int,
        tcp_window: int,
    ):
        self.key            = key               # (src_ip, src_port, dst_ip, dst_port, proto)
        self.dst_port       = key[3]
        self.first_ts       = ts
        self.last_ts        = ts

        self.fwd_lengths: List[int] = [pkt_len]
        self.bwd_lengths: List[int] = []

        self.all_ts:  List[float] = [ts]
        self.fwd_ts:  List[float] = [ts]
        self.bwd_ts:  List[float] = []

        self.fin = self.rst = self.psh = self.ack = self.urg = 0
        self.fwd_psh = self.fwd_urg = 0

        self.init_win_fwd  = tcp_window
        self.init_win_bwd  = -1
        self.fwd_seg_sizes = [pkt_len]

        # Active / idle tracking (in microseconds)
        self.last_active_start = ts
        self.last_active_end   = ts
        self.active_periods: List[Tuple[float, float]] = []
        self.idle_gaps:      List[float] = []

        self._apply_flags(tcp_flags, is_fwd=True)

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _apply_flags(self, flags: int, is_fwd: bool) -> None:
        self.fin += bool(flags & 0x01)
        self.rst += bool(flags & 0x04)
        self.psh += bool(flags & 0x08)
        self.ack += bool(flags & 0x10)
        self.urg += bool(flags & 0x20)
        if is_fwd:
            self.fwd_psh += bool(flags & 0x08)
            self.fwd_urg += bool(flags & 0x20)

    # ── Public API ────────────────────────────────────────────────────────────
    def add_packet(
        self,
        ts: float,
        pkt_len: int,
        is_fwd: bool,
        tcp_flags: int,
        tcp_window: int,
    ) -> None:
        gap_us = (ts - self.last_ts) * 1e6
        if gap_us > ACTIVE_TIMEOUT:
            # Close the current active sub-period
            self.active_periods.append((self.last_active_start, self.last_active_end))
            self.idle_gaps.append(gap_us)
            self.last_active_start = ts
        self.last_active_end = ts
        self.last_ts = ts

        self.all_ts.append(ts)
        if is_fwd:
            self.fwd_lengths.append(pkt_len)
            self.fwd_ts.append(ts)
            self.fwd_seg_sizes.append(pkt_len)
            if self.init_win_fwd < 0:
                self.init_win_fwd = tcp_window
        else:
            self.bwd_lengths.append(pkt_len)
            self.bwd_ts.append(ts)
            if self.init_win_bwd < 0:
                self.init_win_bwd = tcp_window

        self._apply_flags(tcp_flags, is_fwd)

    # ── Feature computation ───────────────────────────────────────────────────
    def to_feature_dict(self) -> Optional[dict]:
        """
        Compute all base features. Returns None for degenerate flows.
        Does NOT include the 14 engineered features — those are added by
        PCAPFeatureExtractor._add_engineered().
        """
        n_fwd = len(self.fwd_lengths)
        n_bwd = len(self.bwd_lengths)
        n_tot = n_fwd + n_bwd

        if n_tot == 0:
            return None

        duration_us = max((self.last_ts - self.first_ts) * 1e6, 1.0)
        duration_s  = duration_us / 1e6

        fwd = np.asarray(self.fwd_lengths, dtype=np.float64)
        bwd = np.asarray(self.bwd_lengths, dtype=np.float64)
        all_pkt = np.concatenate([fwd, bwd])

        # ── IAT arrays (microseconds) ─────────────────────────────────────────
        def iats(ts_list: List[float]) -> np.ndarray:
            if len(ts_list) < 2:
                return np.zeros(1)
            a = np.diff(np.asarray(sorted(ts_list))) * 1e6
            return a if len(a) else np.zeros(1)

        flow_iat = iats(self.all_ts)
        fwd_iat  = iats(self.fwd_ts)
        bwd_iat  = iats(self.bwd_ts)

        # ── Active / idle (microseconds) ──────────────────────────────────────
        self.active_periods.append((self.last_active_start, self.last_active_end))
        act_dur = np.asarray(
            [(e - s) * 1e6 for s, e in self.active_periods if e >= s],
            dtype=np.float64,
        )
        idle_dur = np.asarray(self.idle_gaps, dtype=np.float64)

        # ── Shorthand helpers ─────────────────────────────────────────────────
        def smean(a): return float(np.mean(a)) if len(a) else 0.0
        def sstd(a):  return float(np.std(a))  if len(a) > 1 else 0.0
        def smax(a):  return float(np.max(a))  if len(a) else 0.0
        def smin(a):  return float(np.min(a))  if len(a) else 0.0
        def ssum(a):  return float(np.sum(a))  if len(a) else 0.0
        def svar(a):  return float(np.var(a))  if len(a) > 1 else 0.0

        fwd_bytes = ssum(fwd)
        bwd_bytes = ssum(bwd)

        return {
            # ── 46 base features (exact CICFlowMeter names) ──────────────────
            "Destination Port":              float(self.dst_port),
            "Flow Duration":                 duration_us,
            "Total Fwd Packets":             float(n_fwd),
            "Total Length of Fwd Packets":   fwd_bytes,
            "Fwd Packet Length Max":         smax(fwd) if n_fwd else 0.0,
            "Fwd Packet Length Min":         smin(fwd) if n_fwd else 0.0,
            "Fwd Packet Length Mean":        smean(fwd) if n_fwd else 0.0,
            "Fwd Packet Length Std":         sstd(fwd) if n_fwd else 0.0,
            "Bwd Packet Length Max":         smax(bwd) if n_bwd else 0.0,
            "Bwd Packet Length Min":         smin(bwd) if n_bwd else 0.0,
            "Bwd Packet Length Mean":        smean(bwd) if n_bwd else 0.0,
            "Flow Bytes/s":                  (fwd_bytes + bwd_bytes) / duration_s,
            "Flow Packets/s":                n_tot / duration_s,
            "Flow IAT Mean":                 smean(flow_iat),
            "Flow IAT Std":                  sstd(flow_iat),
            "Flow IAT Max":                  smax(flow_iat),
            "Flow IAT Min":                  smin(flow_iat),
            "Fwd IAT Mean":                  smean(fwd_iat),
            "Fwd IAT Std":                   sstd(fwd_iat),
            "Fwd IAT Min":                   smin(fwd_iat),
            "Bwd IAT Total":                 ssum(bwd_iat),
            "Bwd IAT Mean":                  smean(bwd_iat),
            "Bwd IAT Std":                   sstd(bwd_iat),
            "Bwd IAT Max":                   smax(bwd_iat),
            "Bwd IAT Min":                   smin(bwd_iat),
            "Fwd PSH Flags":                 float(self.fwd_psh),
            "Fwd URG Flags":                 float(self.fwd_urg),
            "Bwd Packets/s":                 n_bwd / duration_s,
            "Min Packet Length":             smin(all_pkt),
            "Max Packet Length":             smax(all_pkt),
            "Packet Length Mean":            smean(all_pkt),
            "Packet Length Variance":        svar(all_pkt),
            "FIN Flag Count":                float(self.fin),
            "RST Flag Count":                float(self.rst),
            "PSH Flag Count":                float(self.psh),
            "ACK Flag Count":                float(self.ack),
            "URG Flag Count":                float(self.urg),
            "Down/Up Ratio":                 float(n_bwd) / float(n_fwd) if n_fwd else 0.0,
            "Init_Win_bytes_forward":        float(max(self.init_win_fwd, 0)),
            "Init_Win_bytes_backward":       float(max(self.init_win_bwd, 0)),
            "min_seg_size_forward":          smin(np.asarray(self.fwd_seg_sizes)) if self.fwd_seg_sizes else 0.0,
            "Active Mean":                   smean(act_dur),
            "Active Std":                    sstd(act_dur),
            "Active Max":                    smax(act_dur),
            "Active Min":                    smin(act_dur),
            "Idle Std":                      sstd(idle_dur),
            # Internal (dropped before model input)
            "_first_ts":                     self.first_ts,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Feature Extractor
# ══════════════════════════════════════════════════════════════════════════════
class PCAPFeatureExtractor:
    """
    Extracts a CICFlowMeter-compatible DataFrame from a PCAP file.
    Output shape: (n_flows, 60) exactly matching feature_list.json ordering.
    """

    def __init__(self, feature_cols: List[str]):
        if not _SCAPY_OK:
            raise RuntimeError(
                "scapy is required for PCAP feature extraction.\n"
                "Install with: pip install scapy"
            )
        self.feature_cols = feature_cols

    # ── Public API ─────────────────────────────────────────────────────────────
    def extract(self, pcap_path: Path, max_flows: int = MAX_FLOWS) -> pd.DataFrame:
        """
        Main entry point.

        Args:
            pcap_path: Path to the .pcap file.
            max_flows: Cap on number of unique flows tracked (prevent OOM).

        Returns:
            DataFrame with exactly len(self.feature_cols) columns, sorted by
            flow start time (ascending).

        Raises:
            ValueError if no flows could be extracted.
            RuntimeError if scapy is not available.
        """
        logger.info(f"[FEATURES] Starting feature extraction: {pcap_path.name}")

        flows: Dict[tuple, _FlowRecord] = {}
        flow_order: List[tuple] = []
        n_pkts = n_skipped = 0

        with PcapReader(str(pcap_path)) as reader:
            for pkt in reader:
                n_pkts += 1
                info = _parse_packet(pkt)
                if info is None:
                    n_skipped += 1
                    continue

                ts, src_ip, src_port, dst_ip, dst_port, proto, pkt_len, flags, window = info
                fwd_key = (src_ip, src_port, dst_ip, dst_port, proto)
                bwd_key = (dst_ip, dst_port, src_ip, src_port, proto)

                if fwd_key in flows:
                    flows[fwd_key].add_packet(ts, pkt_len, True,  flags, window)
                elif bwd_key in flows:
                    flows[bwd_key].add_packet(ts, pkt_len, False, flags, window)
                else:
                    if len(flows) >= max_flows:
                        continue   # hard cap reached — skip new flows
                    flows[fwd_key] = _FlowRecord(fwd_key, ts, pkt_len, flags, window)
                    flow_order.append(fwd_key)

        logger.info(
            f"[FEATURES] Packets: {n_pkts:,} total, {n_skipped:,} skipped, "
            f"{len(flows):,} flows tracked"
        )

        # Finalise flow records → feature dicts
        records: List[dict] = []
        for key in flow_order:
            rec = flows[key]
            feat = rec.to_feature_dict()
            if feat is not None:
                records.append(feat)

        if not records:
            raise ValueError(
                "No valid flows could be extracted from the PCAP. "
                "The capture may contain only non-IP traffic or single-packet flows."
            )

        df = pd.DataFrame(records)
        logger.info(f"[FEATURES] Raw flow records: {len(df):,}")

        # Add 14 engineered features
        df = _add_engineered(df)

        # Sort by flow start time (chronological order for sequence construction)
        if "_first_ts" in df.columns:
            df = df.sort_values("_first_ts").reset_index(drop=True)
            df = df.drop(columns=["_first_ts"])

        # Align to exact model schema
        df = self._align_to_schema(df)

        logger.info(
            f"[FEATURES] Final DataFrame: {len(df):,} flows × {len(df.columns)} features"
        )
        return df

    def validate_schema(self, df: pd.DataFrame) -> dict:
        """
        Validate that *df* exactly matches the model's expected feature schema.
        Returns a diagnostic dict — call this BEFORE scaling.
        """
        expected = self.feature_cols
        got      = list(df.columns)
        missing  = [c for c in expected if c not in got]
        extra    = [c for c in got if c not in expected]
        order_ok = got == expected

        schema_status = "MATCH" if (not missing and not extra and order_ok) else "MISMATCH"
        return {
            "expected":    len(expected),
            "generated":   len(got),
            "missing":     missing,
            "extra":       extra,
            "order_match": order_ok,
            "schema":      schema_status,
        }

    # ── Internal ──────────────────────────────────────────────────────────────
    def _align_to_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Re-order and fill columns to match self.feature_cols exactly.
        Missing columns → 0.0 (with warning).
        Extra columns → dropped silently.
        """
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            logger.warning(
                f"[FEATURES] {len(missing)} feature(s) defaulted to 0.0 "
                f"(not derivable from Zeek/scapy without packet-level data): "
                f"{missing}"
            )
            for col in missing:
                df[col] = 0.0

        result = df[self.feature_cols].copy()
        # Sanitise: replace inf / NaN with 0
        result = result.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        result = result.astype(np.float32)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Helpers — module-level functions
# ══════════════════════════════════════════════════════════════════════════════
def _parse_packet(pkt) -> Optional[tuple]:
    """
    Extract (ts, src_ip, src_port, dst_ip, dst_port, proto, pkt_len, flags, window)
    from a scapy packet.  Returns None for non-IP or malformed packets.
    """
    try:
        if not pkt.haslayer(IP):
            return None

        ip  = pkt[IP]
        ts  = float(pkt.time)
        pkt_len = len(ip)          # IP total length (incl. IP header)

        if pkt.haslayer(TCP):
            tcp    = pkt[TCP]
            return (ts, ip.src, tcp.sport, ip.dst, tcp.dport, 6,
                    pkt_len, int(tcp.flags), int(tcp.window))

        if pkt.haslayer(UDP):
            udp = pkt[UDP]
            return (ts, ip.src, udp.sport, ip.dst, udp.dport, 17,
                    pkt_len, 0, 0)

        # Other IP protocols (ICMP, etc.) — port 0
        return (ts, ip.src, 0, ip.dst, 0, int(ip.proto), pkt_len, 0, 0)

    except Exception:
        return None


def _add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 14 engineered features used during training.
    All derived from the 46 base features already in *df*.
    """
    eps = 1e-9    # avoid div/0

    df["pkt_fwd_bwd_ratio"]   = df["Total Fwd Packets"] / (df["Total Fwd Packets"].shift(0).replace(0, eps) + df.get("Bwd Packets/s", 0).replace(0, eps))
    # Recompute cleanly
    n_fwd = df["Total Fwd Packets"].clip(lower=0)
    n_bwd = (df["Bwd Packets/s"] * (df["Flow Duration"] / 1e6)).clip(lower=0)

    df["pkt_fwd_bwd_ratio"]  = n_fwd / (n_bwd + eps)
    df["byte_fwd_bwd_ratio"] = df["Total Length of Fwd Packets"] / (
        (df["Bwd Packet Length Mean"] * n_bwd).clip(lower=0) + eps
    )
    df["avg_pkt_size_ratio"] = df["Fwd Packet Length Mean"] / (df["Bwd Packet Length Mean"] + eps)
    df["fwd_bwd_iat_ratio"]  = df["Fwd IAT Mean"] / (df["Bwd IAT Mean"] + eps)
    df["active_idle_ratio"]  = df["Active Mean"] / (df["Idle Std"] + eps)
    df["window_size_ratio"]  = df["Init_Win_bytes_forward"] / (df["Init_Win_bytes_backward"] + eps)
    df["flag_density_per_pkt"] = (
        (df["PSH Flag Count"] + df["URG Flag Count"] + df["FIN Flag Count"]) /
        (df["Total Fwd Packets"] + n_bwd + eps)
    )
    df["log1p_flow_duration"]  = np.log1p(df["Flow Duration"].clip(lower=0))
    df["log1p_fwd_pkts"]       = np.log1p(df["Total Fwd Packets"].clip(lower=0))
    df["log1p_bwd_pkts"]       = np.log1p(n_bwd)
    df["log1p_flow_bytes_s"]   = np.log1p(df["Flow Bytes/s"].clip(lower=0))
    df["log1p_pkt_len_mean"]   = np.log1p(df["Packet Length Mean"].clip(lower=0))
    df["log1p_idle_mean"]      = np.log1p(df["Idle Std"].clip(lower=0))   # proxy
    df["log1p_active_mean"]    = np.log1p(df["Active Mean"].clip(lower=0))

    return df
