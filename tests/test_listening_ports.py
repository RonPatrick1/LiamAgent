import socket
import unittest
from unittest import mock

from agent import tools


class ListeningPortsTests(unittest.TestCase):
    def test_decodes_proc_ipv4_loopback(self):
        self.assertEqual(
            tools._decode_proc_net_address(
                "0100007F",
                socket.AF_INET,
            ),
            "127.0.0.1",
        )

    def test_decodes_proc_ipv6_loopback(self):
        self.assertEqual(
            tools._decode_proc_net_address(
                "00000000000000000000000001000000",
                socket.AF_INET6,
            ),
            "::1",
        )

    def test_reports_listeners_and_excludes_used_candidate_ports(self):
        records = [
            ("tcp", "0.0.0.0", 8000),
            ("tcp6", "::", 8002),
        ]

        with mock.patch.object(
            tools,
            "_listening_socket_records",
            return_value=records,
        ):
            result = tools.listening_ports(
                candidate_start=8000,
                candidate_end=8004,
                suggestion_count=3,
            )

        self.assertIn("tcp 0.0.0.0:8000", result)
        self.assertIn("tcp6 :::8002", result)
        self.assertIn("8001, 8003, 8004", result)
        self.assertNotIn(
            "8000, 8001",
            result,
        )

    def test_rejects_privileged_candidate_range(self):
        with self.assertRaisesRegex(
            ValueError,
            "candidate_start must be between 1024 and 65535",
        ):
            tools.listening_ports(
                candidate_start=80,
                candidate_end=90,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
