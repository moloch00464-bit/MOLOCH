#!/usr/bin/env python3
"""
M.O.L.O.C.H. System Health Checker
===================================

Startup self-check and continuous health monitoring.

Checks:
    - Hailo device presence (/dev/hailo0)
    - HEF model files exist
    - Sonoff camera ping
    - PTZ test movement
    - Firmware versions
    - RTSP connection count

Author: M.O.L.O.C.H. System
Date: 2026-02-05
"""

import os
import time
import socket
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class CheckResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealthReport:
    """Full system health report."""
    timestamp: float = field(default_factory=time.time)
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    checks: List[CheckResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        return self.overall_status == HealthStatus.OK

    def summary(self) -> str:
        ok = sum(1 for c in self.checks if c.status == HealthStatus.OK)
        warn = sum(1 for c in self.checks if c.status == HealthStatus.WARNING)
        err = sum(1 for c in self.checks if c.status == HealthStatus.ERROR)
        return f"{ok} OK, {warn} warnings, {err} errors"


class SystemHealthChecker:
    """
    System health checker for startup self-check and monitoring.
    """

    # Hailo device paths
    HAILO_DEVICE = "/dev/hailo0"
    HAILO_FIRMWARE_PATH = "/lib/firmware/hailo"

    # Model paths (secondary SSD)
    MODEL_DIR = Path("/mnt/moloch-data/hailo/models")
    REQUIRED_MODELS = [
        "yolov8m_pose_h10.hef",
        "yolov8s_pose_h10.hef",
        "yolov8m_h10.hef",
    ]

    # Sonoff camera
    SONOFF_IP = os.environ.get("MOLOCH_CAMERA_HOST", "CAMERA_IP")
    SONOFF_RTSP_PORT = 554
    SONOFF_ONVIF_PORT = 8899

    def __init__(self):
        self.last_report: Optional[SystemHealthReport] = None

    def run_startup_checks(self, ptz_controller=None) -> SystemHealthReport:
        """
        Run all startup health checks.

        Args:
            ptz_controller: Optional PTZ controller for movement test

        Returns:
            SystemHealthReport with all check results
        """
        report = SystemHealthReport()
        logger.info("=" * 60)
        logger.info("=== M.O.L.O.C.H. STARTUP SELF-CHECK ===")
        logger.info("=" * 60)

        # Run all checks
        checks = [
            self._check_hailo_device,
            self._check_model_files,
            self._check_sonoff_ping,
            self._check_sonoff_rtsp,
            self._check_sonoff_onvif,
            self._check_rtsp_connections,
            self._check_hailo_firmware,
            self._check_ssd_mount,
        ]

        for check_func in checks:
            try:
                result = check_func()
                report.checks.append(result)

                # Log result
                status_icon = {
                    HealthStatus.OK: "✓",
                    HealthStatus.WARNING: "⚠",
                    HealthStatus.ERROR: "✗",
                    HealthStatus.UNKNOWN: "?"
                }.get(result.status, "?")

                logger.info(f"  {status_icon} {result.name}: {result.message}")

                # Collect errors/warnings
                if result.status == HealthStatus.ERROR:
                    report.errors.append(f"{result.name}: {result.message}")
                elif result.status == HealthStatus.WARNING:
                    report.warnings.append(f"{result.name}: {result.message}")

            except Exception as e:
                error_result = CheckResult(
                    name=check_func.__name__,
                    status=HealthStatus.ERROR,
                    message=f"Check failed: {e}"
                )
                report.checks.append(error_result)
                report.errors.append(f"{check_func.__name__}: {e}")
                logger.error(f"  ✗ {check_func.__name__}: {e}")

        # PTZ test (optional)
        if ptz_controller:
            result = self._check_ptz_movement(ptz_controller)
            report.checks.append(result)
            logger.info(f"  {'✓' if result.status == HealthStatus.OK else '✗'} {result.name}: {result.message}")

        # Determine overall status
        if report.errors:
            report.overall_status = HealthStatus.ERROR
        elif report.warnings:
            report.overall_status = HealthStatus.WARNING
        else:
            report.overall_status = HealthStatus.OK

        logger.info("=" * 60)
        logger.info(f"=== SELF-CHECK COMPLETE: {report.summary()} ===")
        logger.info(f"=== Overall: {report.overall_status.value.upper()} ===")
        logger.info("=" * 60)

        self.last_report = report
        return report

    # === Individual Checks ===

    def _check_hailo_device(self) -> CheckResult:
        """Check if Hailo device exists."""
        start = time.time()

        if os.path.exists(self.HAILO_DEVICE):
            # Check permissions
            readable = os.access(self.HAILO_DEVICE, os.R_OK)
            writable = os.access(self.HAILO_DEVICE, os.W_OK)

            if readable and writable:
                return CheckResult(
                    name="Hailo Device",
                    status=HealthStatus.OK,
                    message=f"{self.HAILO_DEVICE} accessible",
                    duration_ms=(time.time() - start) * 1000
                )
            else:
                return CheckResult(
                    name="Hailo Device",
                    status=HealthStatus.ERROR,
                    message=f"{self.HAILO_DEVICE} permission denied",
                    duration_ms=(time.time() - start) * 1000
                )
        else:
            return CheckResult(
                name="Hailo Device",
                status=HealthStatus.ERROR,
                message=f"{self.HAILO_DEVICE} not found",
                duration_ms=(time.time() - start) * 1000
            )

    def _check_model_files(self) -> CheckResult:
        """Check if required HEF models exist."""
        start = time.time()

        if not self.MODEL_DIR.exists():
            return CheckResult(
                name="Model Files",
                status=HealthStatus.ERROR,
                message=f"Model directory not found: {self.MODEL_DIR}",
                duration_ms=(time.time() - start) * 1000
            )

        missing = []
        found = []
        for model in self.REQUIRED_MODELS:
            path = self.MODEL_DIR / model
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                found.append(f"{model} ({size_mb:.1f}MB)")
            else:
                missing.append(model)

        if missing:
            return CheckResult(
                name="Model Files",
                status=HealthStatus.ERROR,
                message=f"Missing: {', '.join(missing)}",
                duration_ms=(time.time() - start) * 1000,
                details={"missing": missing, "found": found}
            )
        else:
            return CheckResult(
                name="Model Files",
                status=HealthStatus.OK,
                message=f"{len(found)} models found",
                duration_ms=(time.time() - start) * 1000,
                details={"found": found}
            )

    def _check_sonoff_ping(self) -> CheckResult:
        """Ping Sonoff camera."""
        start = time.time()

        try:
            # Use socket to check if host is reachable
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            result = sock.connect_ex((self.SONOFF_IP, 80))
            sock.close()

            if result == 0:
                return CheckResult(
                    name="Sonoff Ping",
                    status=HealthStatus.OK,
                    message=f"{self.SONOFF_IP} reachable",
                    duration_ms=(time.time() - start) * 1000
                )
            else:
                return CheckResult(
                    name="Sonoff Ping",
                    status=HealthStatus.ERROR,
                    message=f"{self.SONOFF_IP} not reachable",
                    duration_ms=(time.time() - start) * 1000
                )

        except Exception as e:
            return CheckResult(
                name="Sonoff Ping",
                status=HealthStatus.ERROR,
                message=f"Ping failed: {e}",
                duration_ms=(time.time() - start) * 1000
            )

    def _check_sonoff_rtsp(self) -> CheckResult:
        """Check Sonoff RTSP port."""
        start = time.time()

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            result = sock.connect_ex((self.SONOFF_IP, self.SONOFF_RTSP_PORT))
            sock.close()

            if result == 0:
                return CheckResult(
                    name="Sonoff RTSP",
                    status=HealthStatus.OK,
                    message=f"Port {self.SONOFF_RTSP_PORT} open",
                    duration_ms=(time.time() - start) * 1000
                )
            else:
                return CheckResult(
                    name="Sonoff RTSP",
                    status=HealthStatus.ERROR,
                    message=f"Port {self.SONOFF_RTSP_PORT} closed",
                    duration_ms=(time.time() - start) * 1000
                )

        except Exception as e:
            return CheckResult(
                name="Sonoff RTSP",
                status=HealthStatus.ERROR,
                message=f"Check failed: {e}",
                duration_ms=(time.time() - start) * 1000
            )

    def _check_sonoff_onvif(self) -> CheckResult:
        """Check Sonoff ONVIF port."""
        start = time.time()

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            result = sock.connect_ex((self.SONOFF_IP, self.SONOFF_ONVIF_PORT))
            sock.close()

            if result == 0:
                return CheckResult(
                    name="Sonoff ONVIF",
                    status=HealthStatus.OK,
                    message=f"Port {self.SONOFF_ONVIF_PORT} open",
                    duration_ms=(time.time() - start) * 1000
                )
            else:
                return CheckResult(
                    name="Sonoff ONVIF",
                    status=HealthStatus.WARNING,
                    message=f"Port {self.SONOFF_ONVIF_PORT} closed (PTZ may fail)",
                    duration_ms=(time.time() - start) * 1000
                )

        except Exception as e:
            return CheckResult(
                name="Sonoff ONVIF",
                status=HealthStatus.WARNING,
                message=f"Check failed: {e}",
                duration_ms=(time.time() - start) * 1000
            )

    def _check_rtsp_connections(self) -> CheckResult:
        """Check for multiple RTSP connections (should be 1 max)."""
        start = time.time()

        try:
            # Count established connections to RTSP port
            result = subprocess.run(
                ["ss", "-tn", f"dst {self.SONOFF_IP}:{self.SONOFF_RTSP_PORT}"],
                capture_output=True, text=True, timeout=5
            )

            # Count ESTABLISHED lines
            lines = result.stdout.strip().split('\n')
            connections = sum(1 for line in lines if 'ESTAB' in line)

            if connections == 0:
                return CheckResult(
                    name="RTSP Connections",
                    status=HealthStatus.OK,
                    message="No existing connections",
                    duration_ms=(time.time() - start) * 1000
                )
            elif connections == 1:
                return CheckResult(
                    name="RTSP Connections",
                    status=HealthStatus.WARNING,
                    message="1 existing connection (may be stale)",
                    duration_ms=(time.time() - start) * 1000
                )
            else:
                return CheckResult(
                    name="RTSP Connections",
                    status=HealthStatus.ERROR,
                    message=f"{connections} connections! Close extras before starting",
                    duration_ms=(time.time() - start) * 1000
                )

        except Exception as e:
            return CheckResult(
                name="RTSP Connections",
                status=HealthStatus.WARNING,
                message=f"Could not check: {e}",
                duration_ms=(time.time() - start) * 1000
            )

    def _check_hailo_firmware(self) -> CheckResult:
        """Check Hailo firmware version."""
        start = time.time()

        try:
            # Try to get firmware info via hailortcli
            result = subprocess.run(
                ["hailortcli", "fw-control", "identify"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                # Parse version from output
                output = result.stdout
                version = "unknown"
                for line in output.split('\n'):
                    if 'Firmware Version' in line or 'Version' in line:
                        version = line.split(':')[-1].strip() if ':' in line else line
                        break

                return CheckResult(
                    name="Hailo Firmware",
                    status=HealthStatus.OK,
                    message=f"Version: {version[:50]}",
                    duration_ms=(time.time() - start) * 1000,
                    details={"full_output": output[:500]}
                )
            else:
                return CheckResult(
                    name="Hailo Firmware",
                    status=HealthStatus.WARNING,
                    message="Could not get firmware info",
                    duration_ms=(time.time() - start) * 1000
                )

        except FileNotFoundError:
            return CheckResult(
                name="Hailo Firmware",
                status=HealthStatus.WARNING,
                message="hailortcli not found",
                duration_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return CheckResult(
                name="Hailo Firmware",
                status=HealthStatus.WARNING,
                message=f"Check failed: {e}",
                duration_ms=(time.time() - start) * 1000
            )

    def _check_ssd_mount(self) -> CheckResult:
        """Check if secondary SSD is mounted."""
        start = time.time()

        mount_path = Path("/mnt/moloch-data")

        if not mount_path.exists():
            return CheckResult(
                name="SSD Mount",
                status=HealthStatus.ERROR,
                message=f"{mount_path} does not exist",
                duration_ms=(time.time() - start) * 1000
            )

        # Check if it's actually mounted (not just directory exists)
        try:
            result = subprocess.run(
                ["mountpoint", "-q", str(mount_path)],
                capture_output=True, timeout=5
            )

            if result.returncode == 0:
                # Get disk space
                statvfs = os.statvfs(mount_path)
                free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)

                return CheckResult(
                    name="SSD Mount",
                    status=HealthStatus.OK,
                    message=f"Mounted, {free_gb:.1f}GB free",
                    duration_ms=(time.time() - start) * 1000
                )
            else:
                return CheckResult(
                    name="SSD Mount",
                    status=HealthStatus.ERROR,
                    message=f"{mount_path} is not a mount point",
                    duration_ms=(time.time() - start) * 1000
                )

        except Exception as e:
            return CheckResult(
                name="SSD Mount",
                status=HealthStatus.WARNING,
                message=f"Could not verify mount: {e}",
                duration_ms=(time.time() - start) * 1000
            )

    def _check_ptz_movement(self, ptz_controller) -> CheckResult:
        """Test PTZ movement."""
        start = time.time()

        try:
            if not ptz_controller:
                return CheckResult(
                    name="PTZ Test",
                    status=HealthStatus.WARNING,
                    message="No PTZ controller provided",
                    duration_ms=(time.time() - start) * 1000
                )

            if not ptz_controller.is_connected:
                return CheckResult(
                    name="PTZ Test",
                    status=HealthStatus.ERROR,
                    message="PTZ not connected",
                    duration_ms=(time.time() - start) * 1000
                )

            # Small test movement: pan right briefly
            logger.info("  ... Testing PTZ (small pan right)")
            result = ptz_controller.continuous_move(0.15, 0.0, timeout_sec=0.5)
            time.sleep(0.5)
            ptz_controller.stop()

            if result:
                return CheckResult(
                    name="PTZ Test",
                    status=HealthStatus.OK,
                    message="Movement test passed",
                    duration_ms=(time.time() - start) * 1000
                )
            else:
                return CheckResult(
                    name="PTZ Test",
                    status=HealthStatus.WARNING,
                    message="Movement command returned False",
                    duration_ms=(time.time() - start) * 1000
                )

        except Exception as e:
            return CheckResult(
                name="PTZ Test",
                status=HealthStatus.ERROR,
                message=f"Test failed: {e}",
                duration_ms=(time.time() - start) * 1000
            )


# === Convenience function ===

def run_startup_checks(ptz_controller=None) -> SystemHealthReport:
    """Run startup health checks."""
    checker = SystemHealthChecker()
    return checker.run_startup_checks(ptz_controller)
