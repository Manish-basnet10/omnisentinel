# cicflow.zeek
# ============
# OmniSentinel custom Zeek policy.
#
# Purpose:
#   Load standard Zeek analysis frameworks so that conn.log is produced
#   in JSON format with all standard fields. CICFlowMeter-compatible
#   per-packet statistics (IAT, packet lengths, TCP flags, window sizes)
#   are computed by the Python feature_service using scapy — Zeek's role
#   here is to (a) validate the PCAP, (b) identify flows & services,
#   and (c) provide a quick sanity-check count of flows.
#
# Usage (called automatically by zeek_service.py):
#   zeek -r input.pcap LogAscii::use_json=T /path/to/cicflow.zeek
#
# Requirements: Zeek >= 7.0

@load base/protocols/conn
@load base/protocols/http
@load base/protocols/dns
@load base/protocols/ftp
@load base/protocols/ssh
@load base/frameworks/notice

# Ensure the conn.log captures all optional fields
redef record Conn::Info += {
    ## Total forward (orig) bytes including IP headers
    orig_ip_bytes:  count &log &optional;
    ## Total backward (resp) bytes including IP headers
    resp_ip_bytes:  count &log &optional;
};
