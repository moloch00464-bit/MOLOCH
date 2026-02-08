#!/usr/bin/env python3
"""
Camera Control Panel - Manual Camera Control GUI
=================================================

Exclusive manual control interface for Sonoff camera.

Features:
- Exclusive lock acquisition (pauses autonomous systems)
- Manual PTZ control with arrow buttons
- Sleep mode control
- LED brightness control
- Night/IR mode control
- Microphone gain control
- Password change
- Status monitoring

Safety:
- Acquires exclusive lock on start
- Releases lock on exit
- Watchdog timer for crash recovery
- Non-blocking async operations

Author: M.O.L.O.C.H. System
Date: 2026-02-07
"""

import sys
import time
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
from typing import Optional
import json

# Setup logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.hardware.unified_camera_controller import (
    UnifiedCameraController,
    ControlMode,
    NightMode,
    LEDLevel
)

# Import autonomy components
try:
    from context.system_autonomy import get_system_autonomy
    AUTONOMY_AVAILABLE = True
except ImportError:
    AUTONOMY_AVAILABLE = False
    logger.warning("SystemAutonomy not available")

try:
    from core.mpo.autonomous_tracker import get_autonomous_tracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    logger.warning("AutonomousTracker not available")


class CameraControlPanel:
    """
    Camera Control Panel GUI.

    Provides exclusive manual control with system pause/resume.
    """

    OWNER_ID = "camera_control_panel"
    WATCHDOG_TIMEOUT = 10.0  # seconds

    def __init__(self, root: tk.Tk):
        """Initialize control panel."""
        self.root = root
        self.root.title("M.O.L.O.C.H. Camera Control Panel")
        self.root.geometry("900x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Camera controller
        self.controller: Optional[UnifiedCameraController] = None
        self.has_exclusive_lock = False

        # Watchdog
        self.watchdog_thread: Optional[threading.Thread] = None
        self.watchdog_active = False
        self.last_heartbeat = time.time()

        # Status update thread
        self.status_thread: Optional[threading.Thread] = None
        self.status_active = False

        # Build UI
        self._build_ui()

        # AUTOMATICALLY pause autonomous systems when panel opens
        self._pause_autonomous_systems()
        logger.info("⚠ Autonomous systems paused automatically on panel startup")

        # Start watchdog
        self._start_watchdog()

        logger.info("Camera Control Panel initialized")

    def _build_ui(self):
        """Build the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        row = 0

        # Title
        title = ttk.Label(main_frame, text="Camera Control Panel",
                         font=('Arial', 16, 'bold'))
        title.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        # =====================================================================
        # Connection Section
        # =====================================================================
        conn_frame = ttk.LabelFrame(main_frame, text="Connection", padding="10")
        conn_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Camera IP
        ttk.Label(conn_frame, text="Camera IP:").grid(row=0, column=0, sticky=tk.W)
        self.camera_ip_var = tk.StringVar(value="192.168.178.25")
        ttk.Entry(conn_frame, textvariable=self.camera_ip_var, width=20).grid(
            row=0, column=1, sticky=tk.W, padx=5)

        # Connect button
        self.connect_btn = ttk.Button(conn_frame, text="Connect Camera",
                                      command=self._connect_camera)
        self.connect_btn.grid(row=0, column=2, padx=5)

        # Disconnect button
        self.disconnect_btn = ttk.Button(conn_frame, text="Disconnect",
                                         command=self._disconnect_camera,
                                         state=tk.DISABLED)
        self.disconnect_btn.grid(row=0, column=3, padx=5)

        # =====================================================================
        # Cloud Login Section
        # =====================================================================
        cloud_frame = ttk.LabelFrame(main_frame, text="Cloud Login", padding="10")
        cloud_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Username
        ttk.Label(cloud_frame, text="Username:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.cloud_username_var = tk.StringVar(value="")
        ttk.Entry(cloud_frame, textvariable=self.cloud_username_var, width=25).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Password
        ttk.Label(cloud_frame, text="Password:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.cloud_password_var = tk.StringVar(value="")
        ttk.Entry(cloud_frame, textvariable=self.cloud_password_var, width=25,
                 show='*').grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Login button
        self.cloud_login_btn = ttk.Button(cloud_frame, text="Login",
                                         command=self._cloud_login)
        self.cloud_login_btn.grid(row=0, column=2, rowspan=2, padx=5)

        # Save to Config button
        self.cloud_save_btn = ttk.Button(cloud_frame, text="Save to Config",
                                        command=self._save_cloud_config,
                                        state=tk.DISABLED)
        self.cloud_save_btn.grid(row=0, column=3, rowspan=2, padx=5)

        # Response Status Labels
        self.cloud_response_code = ttk.Label(cloud_frame, text="Response: -", foreground="gray")
        self.cloud_response_code.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.cloud_token_status = ttk.Label(cloud_frame, text="Token: -", foreground="gray")
        self.cloud_token_status.grid(row=2, column=2, columnspan=2, sticky=tk.W, pady=2)

        # Load cloud config on init
        self._load_cloud_config_ui()

        # =====================================================================
        # Status Section
        # =====================================================================
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Status labels
        self.status_onvif = ttk.Label(status_frame, text="ONVIF: Disconnected",
                                     foreground="red")
        self.status_onvif.grid(row=0, column=0, padx=10, sticky=tk.W)

        self.status_cloud = ttk.Label(status_frame, text="Cloud: Disconnected",
                                     foreground="red")
        self.status_cloud.grid(row=0, column=1, padx=10, sticky=tk.W)

        self.status_lock = ttk.Label(status_frame, text="Exclusive Lock: Inactive",
                                    foreground="orange")
        self.status_lock.grid(row=0, column=2, padx=10, sticky=tk.W)

        # Position display
        self.position_label = ttk.Label(status_frame, text="Position: N/A")
        self.position_label.grid(row=1, column=0, columnspan=3, pady=5)

        # =====================================================================
        # PTZ Control Section
        # =====================================================================
        ptz_frame = ttk.LabelFrame(main_frame, text="PTZ Control", padding="10")
        ptz_frame.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)

        # Arrow buttons
        ttk.Button(ptz_frame, text="↑", width=5, command=lambda: self._ptz_move("up")).grid(
            row=0, column=1, padx=2, pady=2)
        ttk.Button(ptz_frame, text="←", width=5, command=lambda: self._ptz_move("left")).grid(
            row=1, column=0, padx=2, pady=2)
        ttk.Button(ptz_frame, text="⌂", width=5, command=self._ptz_home).grid(
            row=1, column=1, padx=2, pady=2)
        ttk.Button(ptz_frame, text="→", width=5, command=lambda: self._ptz_move("right")).grid(
            row=1, column=2, padx=2, pady=2)
        ttk.Button(ptz_frame, text="↓", width=5, command=lambda: self._ptz_move("down")).grid(
            row=2, column=1, padx=2, pady=2)

        # Stop button
        ttk.Button(ptz_frame, text="STOP", width=15, command=self._ptz_stop).grid(
            row=3, column=0, columnspan=3, pady=10)

        # =====================================================================
        # Camera Features Section
        # =====================================================================
        features_frame = ttk.LabelFrame(main_frame, text="Camera Features", padding="10")
        features_frame.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S),
                          pady=5, padx=5)
        row += 1

        # Sleep Mode
        ttk.Label(features_frame, text="Sleep/Privacy Mode:").grid(
            row=0, column=0, sticky=tk.W, pady=5)
        sleep_btns = ttk.Frame(features_frame)
        sleep_btns.grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Button(sleep_btns, text="ON", width=8,
                  command=lambda: self._set_sleep(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(sleep_btns, text="OFF", width=8,
                  command=lambda: self._set_sleep(False)).pack(side=tk.LEFT, padx=2)

        # LED Control
        ttk.Label(features_frame, text="LED Brightness:").grid(
            row=1, column=0, sticky=tk.W, pady=5)
        self.led_var = tk.StringVar(value="OFF")
        led_combo = ttk.Combobox(features_frame, textvariable=self.led_var,
                                values=["OFF", "LOW", "MEDIUM", "HIGH"],
                                width=15, state="readonly")
        led_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        led_combo.bind("<<ComboboxSelected>>", self._on_led_change)

        # Night Mode
        ttk.Label(features_frame, text="Night/IR Mode:").grid(
            row=2, column=0, sticky=tk.W, pady=5)
        self.night_var = tk.StringVar(value="AUTO")
        night_combo = ttk.Combobox(features_frame, textvariable=self.night_var,
                                   values=["AUTO", "DAY", "NIGHT"],
                                   width=15, state="readonly")
        night_combo.grid(row=2, column=1, sticky=tk.W, pady=5)
        night_combo.bind("<<ComboboxSelected>>", self._on_night_change)

        # Mic Gain
        ttk.Label(features_frame, text="Mic Gain:").grid(
            row=3, column=0, sticky=tk.W, pady=5)
        self.mic_gain_var = tk.DoubleVar(value=70.0)
        mic_scale = ttk.Scale(features_frame, from_=0, to=100,
                             variable=self.mic_gain_var, orient=tk.HORIZONTAL,
                             length=200, command=self._on_mic_gain_change)
        mic_scale.grid(row=3, column=1, sticky=tk.W, pady=5)
        self.mic_gain_label = ttk.Label(features_frame, text="70%")
        self.mic_gain_label.grid(row=3, column=2, pady=5, padx=5)

        # =====================================================================
        # Log Section
        # =====================================================================
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S),
                      pady=5)
        row += 1

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=80,
                                                  state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Configure weights
        main_frame.rowconfigure(row-1, weight=1)

    # =========================================================================
    # Connection Management
    # =========================================================================

    def _connect_camera(self):
        """Connect to camera and acquire exclusive lock."""
        self._log("Connecting to camera...")

        try:
            # Create controller
            self.controller = UnifiedCameraController(
                camera_ip=self.camera_ip_var.get()
            )

            # Connect
            if not self.controller.connect():
                self._log("ERROR: Connection failed", "red")
                messagebox.showerror("Connection Error", "Failed to connect to camera")
                return

            self._log("✓ Camera connected", "green")

            # Acquire exclusive lock
            if self.controller.acquire_exclusive(self.OWNER_ID):
                self.has_exclusive_lock = True
                self._log("✓ Exclusive lock acquired", "green")
                # Systems already paused in __init__, no need to pause again
            else:
                self._log("⚠ Failed to acquire exclusive lock", "orange")
                messagebox.showwarning("Lock Warning",
                                     f"Camera is controlled by: {self.controller.exclusive_owner}")

            # Update UI
            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)

            # Start status updates
            self._start_status_updates()

        except Exception as e:
            self._log(f"ERROR: {e}", "red")
            messagebox.showerror("Error", str(e))

    def _disconnect_camera(self):
        """Disconnect camera and release lock."""
        if not self.controller:
            return

        self._log("Disconnecting...")

        # Stop status updates
        self._stop_status_updates()

        # Release lock
        if self.has_exclusive_lock:
            self.controller.release_exclusive(self.OWNER_ID)
            self.has_exclusive_lock = False
            self._log("✓ Exclusive lock released", "green")
            self._resume_autonomous_systems()

        # Disconnect
        self.controller.disconnect()
        self.controller = None

        # Update UI
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)

        self._log("✓ Disconnected", "green")

    # =========================================================================
    # PTZ Control
    # =========================================================================

    def _ptz_move(self, direction: str):
        """Move PTZ in direction."""
        if not self.controller or not self.has_exclusive_lock:
            self._log("⚠ No exclusive lock", "orange")
            return

        step = 10.0

        try:
            if direction == "up":
                self.controller.move_relative(0, step)
            elif direction == "down":
                self.controller.move_relative(0, -step)
            elif direction == "left":
                self.controller.move_relative(step, 0)
            elif direction == "right":
                self.controller.move_relative(-step, 0)

            self._log(f"PTZ: {direction}")
        except Exception as e:
            self._log(f"ERROR: PTZ move failed: {e}", "red")

    def _ptz_home(self):
        """Move PTZ to home position."""
        if not self.controller or not self.has_exclusive_lock:
            self._log("⚠ No exclusive lock", "orange")
            return

        try:
            self.controller.goto_home()
            self._log("PTZ: Home")
        except Exception as e:
            self._log(f"ERROR: PTZ home failed: {e}", "red")

    def _ptz_stop(self):
        """Stop PTZ movement."""
        if not self.controller:
            return

        try:
            self.controller.stop()
            self._log("PTZ: Stop")
        except Exception as e:
            self._log(f"ERROR: PTZ stop failed: {e}", "red")

    # =========================================================================
    # Camera Features
    # =========================================================================

    def _set_sleep(self, enabled: bool):
        """Set sleep/privacy mode."""
        if not self.controller:
            self._log("⚠ Not connected", "orange")
            return

        action = "ON" if enabled else "OFF"
        self._log(f"Setting sleep mode {action}...")

        def task():
            try:
                result = self.controller.set_sleep_mode(enabled)
                if result:
                    self._log(f"✓ Sleep mode {action}", "green")
                else:
                    self._log(f"✗ Sleep mode {action} failed (not available)", "orange")
            except Exception as e:
                self._log(f"ERROR: {e}", "red")

        threading.Thread(target=task, daemon=True).start()

    def _on_led_change(self, event):
        """Handle LED level change."""
        if not self.controller:
            return

        level_str = self.led_var.get()
        level_map = {"OFF": LEDLevel.OFF, "LOW": LEDLevel.LOW,
                    "MEDIUM": LEDLevel.MEDIUM, "HIGH": LEDLevel.HIGH}
        level = level_map[level_str]

        self._log(f"Setting LED to {level_str}...")

        def task():
            try:
                result = self.controller.set_led_level(level)
                if result:
                    self._log(f"✓ LED set to {level_str}", "green")
                else:
                    self._log(f"✗ LED control failed (not available)", "orange")
            except Exception as e:
                self._log(f"ERROR: {e}", "red")

        threading.Thread(target=task, daemon=True).start()

    def _on_night_change(self, event):
        """Handle night mode change."""
        if not self.controller:
            return

        mode_str = self.night_var.get()
        mode_map = {"AUTO": NightMode.AUTO, "DAY": NightMode.DAY, "NIGHT": NightMode.NIGHT}
        mode = mode_map[mode_str]

        self._log(f"Setting night mode to {mode_str}...")

        def task():
            try:
                result = self.controller.set_night_mode(mode)
                if result:
                    self._log(f"✓ Night mode set to {mode_str}", "green")
                else:
                    self._log(f"✗ Night mode control failed (not available)", "orange")
            except Exception as e:
                self._log(f"ERROR: {e}", "red")

        threading.Thread(target=task, daemon=True).start()

    def _on_mic_gain_change(self, value):
        """Handle mic gain change."""
        gain = float(value) / 100.0
        self.mic_gain_label.config(text=f"{int(float(value))}%")

        if not self.controller:
            return

        def task():
            try:
                result = self.controller.set_mic_gain(gain)
                if not result:
                    self._log(f"✗ Mic gain control failed (not available)", "orange")
            except Exception as e:
                self._log(f"ERROR: {e}", "red")

        threading.Thread(target=task, daemon=True).start()

    # =========================================================================
    # Cloud Login/Config
    # =========================================================================

    def _load_cloud_config_ui(self):
        """Load cloud config and populate UI fields."""
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "camera_cloud.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    cloud_config = config.get("cloud_config", {})
                    self.cloud_username_var.set(cloud_config.get("username", ""))
                    self.cloud_password_var.set(cloud_config.get("password", ""))
                    # Don't log here - log_text not created yet
        except Exception as e:
            # Don't log here - log_text not created yet
            pass

    def _cloud_login(self):
        """Test cloud login credentials with eWeLink API v2."""
        username = self.cloud_username_var.get()
        password = self.cloud_password_var.get()

        if not username or not password:
            self._log("⚠ Username und Password erforderlich", "orange")
            return

        self._log(f"eWeLink Login: {username}...")

        def task():
            import asyncio
            try:
                # Create cloud bridge with credentials
                from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig
                import aiohttp

                config_path = Path(__file__).parent.parent.parent / "config" / "camera_cloud.json"
                with open(config_path, 'r') as f:
                    cloud_data = json.load(f)
                    cloud_cfg = cloud_data.get("cloud_config", {})

                config = CloudConfig(
                    enabled=True,
                    api_base_url=cloud_cfg.get("api_base_url", "https://eu-api.coolkit.cc"),
                    app_id=cloud_cfg.get("app_id", ""),
                    app_secret=cloud_cfg.get("app_secret", ""),
                    username=username,
                    password=password
                )

                bridge = CameraCloudBridge(config)

                async def do_login():
                    bridge.session = aiohttp.ClientSession()
                    try:
                        result = await bridge.ewelink_login()

                        # Update GUI with response
                        status_code = result['status_code']
                        success = result['success']
                        error = result['error_message']
                        token = result.get('token', '')

                        # Log response details
                        self._log(f"HTTP {status_code}", "green" if success else "red")

                        if success:
                            self._log(f"✓ Login erfolgreich", "green")
                            self._log(f"  Token: {token[:20]}...", "green")
                            self.root.after(0, lambda: self.cloud_save_btn.config(state=tk.NORMAL))
                            self.root.after(0, lambda: self.cloud_response_code.config(
                                text=f"Response: {status_code}", foreground="green"))
                            self.root.after(0, lambda: self.cloud_token_status.config(
                                text=f"Token: {token[:16]}...", foreground="green"))
                        else:
                            self._log(f"✗ Login fehlgeschlagen: {error}", "red")
                            self.root.after(0, lambda: self.cloud_response_code.config(
                                text=f"Response: {status_code}", foreground="red"))
                            self.root.after(0, lambda: self.cloud_token_status.config(
                                text="Token: Failed", foreground="red"))

                        # Log full response body for debugging
                        body = result.get('response_body', {})
                        self._log(f"  Body: {json.dumps(body, indent=2)[:200]}", "gray")

                    finally:
                        await bridge.session.close()

                asyncio.run(do_login())

            except Exception as e:
                self._log(f"ERROR: {e}", "red")
                import traceback
                self._log(f"  {traceback.format_exc()[:200]}", "red")

        threading.Thread(target=task, daemon=True).start()

    def _save_cloud_config(self):
        """Save cloud credentials to config file."""
        username = self.cloud_username_var.get()
        password = self.cloud_password_var.get()

        if not username or not password:
            self._log("⚠ Username und Password erforderlich", "orange")
            return

        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "camera_cloud.json"

            # Load existing config
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {"cloud_enabled": True, "cloud_config": {}}

            # Update credentials
            cloud_config = config.get("cloud_config", {})
            cloud_config["username"] = username
            cloud_config["password"] = password
            config["cloud_config"] = cloud_config

            # Save
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            self._log("✓ Cloud Config gespeichert", "green")
            self.cloud_save_btn.config(state=tk.DISABLED)

        except Exception as e:
            self._log(f"ERROR: {e}", "red")

    # =========================================================================
    # System Pause/Resume
    # =========================================================================

    def _pause_autonomous_systems(self):
        """Pause all autonomous systems using SystemAutonomy."""
        self._log("Pausiere autonome Systeme...")

        # Use SystemAutonomy for centralized control
        if AUTONOMY_AVAILABLE:
            try:
                autonomy = get_system_autonomy()
                autonomy.set_user_override(True)
                autonomy.set_autonomous_mode(False)  # WICHTIG: Autonomous Mode deaktivieren!
                autonomy.stop_vision()
                self._log("  ✓ SystemAutonomy: User Override aktiv")
                self._log("  ✓ Autonomous Mode deaktiviert")
                self._log("  ✓ Vision gestoppt")
            except Exception as e:
                self._log(f"  ⚠ SystemAutonomy Fehler: {e}", "orange")

        # Stop tracker explicitly
        if TRACKER_AVAILABLE:
            try:
                tracker = get_autonomous_tracker()
                tracker.stop()
                self._log("  ✓ AutonomousTracker gestoppt")
            except Exception as e:
                self._log(f"  ⚠ Tracker Fehler: {e}", "orange")

        self._log("✓ Autonome Systeme pausiert", "green")

    def _resume_autonomous_systems(self):
        """Resume all autonomous systems."""
        self._log("Aktiviere autonome Systeme...")

        # Resume SystemAutonomy
        if AUTONOMY_AVAILABLE:
            try:
                autonomy = get_system_autonomy()
                autonomy.set_user_override(False)
                autonomy.set_autonomous_mode(True)  # Autonomous Mode wieder aktivieren
                autonomy.start_vision()
                self._log("  ✓ SystemAutonomy: User Override deaktiviert")
                self._log("  ✓ Autonomous Mode aktiviert")
                self._log("  ✓ Vision gestartet")
            except Exception as e:
                self._log(f"  ⚠ SystemAutonomy Fehler: {e}", "orange")

        # Restart tracker
        if TRACKER_AVAILABLE:
            try:
                tracker = get_autonomous_tracker()
                tracker.start()
                self._log("  ✓ AutonomousTracker gestartet")
            except Exception as e:
                self._log(f"  ⚠ Tracker Fehler: {e}", "orange")

        self._log("✓ Autonome Systeme aktiviert", "green")

    # =========================================================================
    # Status Updates
    # =========================================================================

    def _start_status_updates(self):
        """Start status update thread."""
        self.status_active = True
        self.status_thread = threading.Thread(target=self._status_update_loop, daemon=True)
        self.status_thread.start()

    def _stop_status_updates(self):
        """Stop status update thread."""
        self.status_active = False
        if self.status_thread:
            self.status_thread.join(timeout=1.0)

    def _status_update_loop(self):
        """Status update loop."""
        while self.status_active and self.controller:
            try:
                status = self.controller.get_status()

                # Update status labels
                self.root.after(0, self._update_status_labels, status)

                # Heartbeat
                self.last_heartbeat = time.time()

            except Exception as e:
                logger.error(f"Status update error: {e}")

            time.sleep(1.0)

    def _update_status_labels(self, status):
        """Update status labels in GUI thread."""
        # ONVIF
        if status.connected:
            self.status_onvif.config(text="ONVIF: Connected", foreground="green")
        else:
            self.status_onvif.config(text="ONVIF: Disconnected", foreground="red")

        # Cloud
        if status.cloud_connected:
            self.status_cloud.config(text="Cloud: Connected", foreground="green")
        else:
            self.status_cloud.config(text="Cloud: Disconnected", foreground="red")

        # Lock
        if self.has_exclusive_lock:
            self.status_lock.config(text="Exclusive Lock: Active", foreground="green")
        else:
            self.status_lock.config(text="Exclusive Lock: Inactive", foreground="orange")

        # Position
        if status.position:
            self.position_label.config(
                text=f"Position: Pan={status.position.pan:.1f}°, Tilt={status.position.tilt:.1f}°"
            )

    # =========================================================================
    # Watchdog
    # =========================================================================

    def _start_watchdog(self):
        """Start watchdog thread."""
        self.watchdog_active = True
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()
        logger.info("Watchdog started")

    def _watchdog_loop(self):
        """Watchdog loop - releases lock if GUI becomes unresponsive."""
        while self.watchdog_active:
            time.sleep(1.0)

            if self.has_exclusive_lock and self.controller:
                # Check if heartbeat is stale
                elapsed = time.time() - self.last_heartbeat
                if elapsed > self.WATCHDOG_TIMEOUT:
                    logger.warning(f"Watchdog triggered! No heartbeat for {elapsed:.1f}s")
                    try:
                        self.controller.release_exclusive(self.OWNER_ID)
                        self.controller.set_mode(ControlMode.AUTONOMOUS)
                        logger.info("Watchdog released exclusive lock")
                    except:
                        pass
                    break

    # =========================================================================
    # Logging
    # =========================================================================

    def _log(self, message: str, color: str = "black"):
        """Add message to log."""
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}\n"

        # Log to logger always
        logger.info(message)

        # Only update GUI if log_text exists
        if not hasattr(self, 'log_text'):
            return

        self.log_text.config(state=tk.NORMAL)

        # Insert with color
        if color != "black":
            tag = f"color_{color}"
            self.log_text.tag_config(tag, foreground=color)
            self.log_text.insert(tk.END, log_msg, tag)
        else:
            self.log_text.insert(tk.END, log_msg)

        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def on_closing(self):
        """Handle window close."""
        self._log("Closing...")

        # Disconnect
        if self.controller:
            self._disconnect_camera()

        # Stop watchdog
        self.watchdog_active = False

        # Close window
        self.root.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = CameraControlPanel(root)
    root.mainloop()


if __name__ == "__main__":
    main()
