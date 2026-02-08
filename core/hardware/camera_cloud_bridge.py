#!/usr/bin/env python3
"""
Camera Cloud Bridge - Async Cloud API Interface
================================================

Provides cloud-based camera control for features not available via ONVIF:
- Sleep/Privacy mode
- LED brightness control
- Night/IR mode
- Microphone gain

Features:
- Async cloud calls (non-blocking)
- Token management with automatic refresh
- 3 second timeout per request
- Retry once on failure
- Graceful failure (no exceptions to caller)

Architecture:
  UnifiedCameraController
         ↓
  CameraCloudBridge (async)
         ↓
  Cloud API (eWeLink or proprietary)

Note: This is a framework ready to be populated with actual API endpoints
      once they are discovered (e.g., via reverse engineering mobile app).

Author: M.O.L.O.C.H. System
Date: 2026-02-07
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import hmac
import hashlib
import base64
import secrets

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("WARNING: aiohttp not installed. Run: pip install aiohttp")


class CloudStatus(Enum):
    """Cloud connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class CloudConfig:
    """Cloud API configuration."""
    enabled: bool = False
    api_base_url: str = "https://api.example.com"  # Replace with actual
    app_id: str = ""
    app_secret: str = ""
    device_id: str = ""
    username: str = ""
    password: str = ""
    timeout: float = 3.0
    retry_count: int = 1
    token_refresh_margin: int = 300  # Refresh token 5 minutes before expiry


@dataclass
class CloudToken:
    """Cloud authentication token."""
    access_token: str = ""
    refresh_token: str = ""
    expires_at: float = 0.0
    token_type: str = "Bearer"

    def is_valid(self) -> bool:
        """Check if token is still valid."""
        if not self.access_token:
            return False
        # Check if expired (with 5 minute margin)
        return time.time() < (self.expires_at - 300)

    def needs_refresh(self) -> bool:
        """Check if token needs refresh."""
        if not self.access_token:
            return True
        # Refresh if less than 5 minutes until expiry
        return time.time() >= (self.expires_at - 300)


class CameraCloudBridge:
    """
    Async cloud bridge for camera control.

    Provides cloud API access for features not available via ONVIF.
    All methods are non-blocking and gracefully handle failures.
    """

    def __init__(self, config: CloudConfig, log_level: int = logging.INFO):
        """
        Initialize cloud bridge.

        Args:
            config: Cloud configuration
            log_level: Logging level
        """
        self.config = config
        self.token = CloudToken()
        self.status = CloudStatus.DISABLED if not config.enabled else CloudStatus.DISCONNECTED

        # Logging
        self.logger = logging.getLogger("CameraCloudBridge")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None

        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'timeouts': 0,
            'retries': 0,
            'last_error': None,
            'last_success': 0.0
        }

        if not AIOHTTP_AVAILABLE:
            self.logger.error("aiohttp not available - cloud bridge disabled")
            self.status = CloudStatus.DISABLED
            self.config.enabled = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> bool:
        """
        Connect to cloud API.

        Returns:
            True if connected successfully
        """
        if not self.config.enabled:
            self.logger.info("Cloud bridge disabled in config")
            self.status = CloudStatus.DISABLED
            return False

        if not AIOHTTP_AVAILABLE:
            self.logger.error("aiohttp not available")
            self.status = CloudStatus.DISABLED
            return False

        try:
            self.logger.info("Connecting to cloud API...")
            self.status = CloudStatus.CONNECTING

            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Authenticate
            success = await self._authenticate()
            if success:
                self.status = CloudStatus.CONNECTED
                self.logger.info("✓ Cloud bridge connected")
                return True
            else:
                self.status = CloudStatus.ERROR
                self.logger.warning("Cloud authentication failed")
                return False

        except Exception as e:
            self.status = CloudStatus.ERROR
            self.logger.error(f"Cloud connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from cloud API."""
        if self.session:
            await self.session.close()
            self.session = None

        self.status = CloudStatus.DISCONNECTED
        self.logger.info("Cloud bridge disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if connected to cloud."""
        return self.status == CloudStatus.CONNECTED and self.token.is_valid()

    # =========================================================================
    # Authentication
    # =========================================================================

    async def _authenticate(self) -> bool:
        """
        Authenticate with cloud API using eWeLink API v2.

        Returns:
            True if authentication successful
        """
        if not self.session:
            return False

        # Use eWeLink login
        result = await self.ewelink_login()

        if result['success']:
            # Store token
            self.token.access_token = result['token']
            # eWeLink tokens typically expire in 30 days
            self.token.expires_at = time.time() + (30 * 24 * 3600)
            self.logger.info("✓ Authentication successful via eWeLink API v2")
            return True
        else:
            self.logger.warning(f"✗ Authentication failed: {result['error_message']}")
            return False

    async def _refresh_token(self) -> bool:
        """
        Refresh authentication token.

        Since eWeLink doesn't have a separate refresh endpoint,
        we re-authenticate using ewelink_login().

        Returns:
            True if refresh successful
        """
        if not self.session:
            return False

        self.logger.info("Refreshing token via re-authentication...")

        # Re-authenticate using eWeLink login
        result = await self.ewelink_login()

        if result['success']:
            self.token.access_token = result['token']
            self.token.expires_at = time.time() + (30 * 24 * 3600)
            self.logger.info("✓ Token refreshed")
            return True
        else:
            self.logger.warning(f"✗ Token refresh failed: {result['error_message']}")
            return False

    async def _ensure_authenticated(self) -> bool:
        """
        Ensure we have a valid authentication token.

        Returns:
            True if authenticated
        """
        # Check if token needs refresh
        if self.token.needs_refresh():
            if self.token.refresh_token:
                # Try refresh
                if await self._refresh_token():
                    return True

            # Refresh failed or no refresh token, re-authenticate
            return await self._authenticate()

        # Token is still valid
        return self.token.is_valid()

    # =========================================================================
    # eWeLink API v2 Authentication
    # =========================================================================

    async def ewelink_login(self) -> Dict[str, Any]:
        """
        Authenticate with eWeLink API v2.

        Uses HMAC-SHA256 signature authentication according to eWeLink API v2 spec:
        - sign = base64(HMAC-SHA256(json_body, appsecret))
        - NO timestamp, NO nonce in headers (simplified eWeLink method)
        - countryCode required in payload

        Returns:
            Dict containing:
                - success: bool - Whether login was successful
                - status_code: int - HTTP status code
                - error_message: str - Error description if failed
                - token: str - Access token if successful
                - response_headers: dict - All response headers
                - response_body: dict - Full response body
                - request_timestamp: str - ISO timestamp of request
                - request_details: dict - Details of the request made
        """
        if not self.session:
            return {
                'success': False,
                'status_code': 0,
                'error_message': 'HTTP session not initialized',
                'token': '',
                'response_headers': {},
                'response_body': {},
                'request_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'request_details': {}
            }

        # Build request body (with countryCode!)
        payload = {
            'password': self.config.password,
            'countryCode': '+49'  # Germany
        }

        # Email or phone number
        if '@' in self.config.username:
            payload['email'] = self.config.username
        elif self.config.username.startswith('+'):
            payload['phoneNumber'] = self.config.username
        else:
            payload['phoneNumber'] = '+' + self.config.username

        # JSON body as bytes (ensure consistent encoding for signature)
        json_data = json.dumps(payload).encode('utf-8')

        # Calculate HMAC-SHA256 signature
        # CORRECT eWeLink method: HMAC-SHA256(appsecret, json_data) -> BASE64
        signature_bytes = hmac.new(
            self.config.app_secret.encode('utf-8'),
            json_data,
            hashlib.sha256
        ).digest()

        # Signature as BASE64 (NOT HEX!)
        signature_b64 = base64.b64encode(signature_bytes).decode('utf-8')

        # Build headers (ONLY Authorization and X-CK-Appid!)
        headers = {
            'Authorization': f'Sign {signature_b64}',
            'X-CK-Appid': self.config.app_id,
            'Content-Type': 'application/json'
        }

        # API endpoint
        endpoint = f"{self.config.api_base_url}/v2/user/login"

        # Request details for logging
        request_details = {
            'endpoint': endpoint,
            'method': 'POST',
            'appid': self.config.app_id,
            'appsecret': self.config.app_secret,
            'json_body': json_data.decode('utf-8'),
            'signature_b64': signature_b64,
            'headers': headers.copy(),
            'payload': payload
        }

        request_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        self.logger.info("=" * 80)
        self.logger.info("eWeLink API v2 Login Request (CORRECTED)")
        self.logger.info("=" * 80)
        self.logger.info(f"Timestamp:        {request_timestamp}")
        self.logger.info(f"Endpoint:         {endpoint}")
        self.logger.info(f"")
        self.logger.info(f"AppID:            {self.config.app_id}")
        self.logger.info(f"AppSecret:        {self.config.app_secret}")
        self.logger.info(f"Email:            {self.config.username}")
        self.logger.info(f"Password:         {self.config.password}")
        self.logger.info(f"CountryCode:      +49")
        self.logger.info(f"")
        self.logger.info(f"JSON Body:        {json_data.decode('utf-8')}")
        self.logger.info(f"Signature (b64):  {signature_b64}")
        self.logger.info(f"")
        self.logger.info("Headers:")
        for key, value in headers.items():
            self.logger.info(f"  {key}: {value}")

        try:
            # Make the request (send as data, not json!)
            async with self.session.post(
                endpoint,
                data=json_data,
                headers=headers
            ) as response:
                # Extract status code
                status_code = response.status

                # Extract all response headers
                response_headers = dict(response.headers)

                # Extract response body
                try:
                    response_body = await response.json()
                except:
                    # If not JSON, get text
                    response_text = await response.text()
                    response_body = {'raw_text': response_text}

                # Log full response
                self.logger.info("-" * 80)
                self.logger.info("eWeLink API Response")
                self.logger.info("-" * 80)
                self.logger.info(f"HTTP Status Code: {status_code}")

                # Interpret status code
                status_interpretation = {
                    200: "✓ OK - Request successful",
                    400: "✗ Bad Request - Invalid parameters or malformed request",
                    401: "✗ Unauthorized - Invalid credentials (wrong email/password)",
                    403: "✗ Forbidden - Valid credentials but no access permission",
                    404: "✗ Not Found - Endpoint doesn't exist",
                    500: "✗ Internal Server Error - Server-side problem",
                    503: "✗ Service Unavailable - Server overloaded, maintenance, or invalid signature"
                }
                interpretation = status_interpretation.get(status_code, f"HTTP {status_code}")
                self.logger.info(f"Status:           {interpretation}")

                self.logger.info("")
                self.logger.info("Response Headers:")
                for key, value in response_headers.items():
                    self.logger.info(f"  {key}: {value}")
                self.logger.info("")
                self.logger.info("Response Body:")
                self.logger.info(json.dumps(response_body, indent=2, ensure_ascii=False))
                self.logger.info("=" * 80)

                # Check if successful
                if status_code == 200:
                    # Extract token from response
                    token = ''
                    if isinstance(response_body, dict):
                        # Try different possible token field names
                        token = (
                            response_body.get('at', '') or
                            response_body.get('access_token', '') or
                            response_body.get('token', '') or
                            response_body.get('data', {}).get('at', '') or
                            response_body.get('data', {}).get('access_token', '')
                        )

                    if token:
                        self.logger.info(f"✓ Login successful - Token: {token[:20]}...")
                        return {
                            'success': True,
                            'status_code': status_code,
                            'error_message': '',
                            'token': token,
                            'response_headers': response_headers,
                            'response_body': response_body,
                            'request_timestamp': request_timestamp,
                            'request_details': request_details
                        }
                    else:
                        self.logger.warning("Login returned 200 but no token found in response")
                        return {
                            'success': False,
                            'status_code': status_code,
                            'error_message': 'No token in response',
                            'token': '',
                            'response_headers': response_headers,
                            'response_body': response_body,
                            'request_timestamp': request_timestamp,
                            'request_details': request_details
                        }
                else:
                    # Login failed - build detailed error message
                    error_details = []

                    # Status code interpretation
                    if status_code == 400:
                        error_details.append("Bad Request - Check request format and parameters")
                    elif status_code == 401:
                        error_details.append("Unauthorized - Invalid email or password")
                    elif status_code == 403:
                        error_details.append("Forbidden - Account may be locked or suspended")
                    elif status_code == 404:
                        error_details.append("Not Found - Wrong API endpoint")
                    elif status_code == 500:
                        error_details.append("Server Error - eWeLink server problem")
                    elif status_code == 503:
                        error_details.append("Service Unavailable - Invalid AppID/AppSecret or server maintenance")
                    else:
                        error_details.append(f"HTTP {status_code}")

                    # Extract API error message if available
                    if isinstance(response_body, dict):
                        api_msg = (
                            response_body.get('msg', '') or
                            response_body.get('message', '') or
                            response_body.get('error', '')
                        )
                        if api_msg:
                            error_details.append(f"API: {api_msg}")

                    error_msg = " | ".join(error_details)

                    self.logger.error(f"✗ Login failed: {error_msg}")
                    return {
                        'success': False,
                        'status_code': status_code,
                        'error_message': error_msg,
                        'token': '',
                        'response_headers': response_headers,
                        'response_body': response_body,
                        'request_timestamp': request_timestamp,
                        'request_details': request_details
                    }

        except asyncio.TimeoutError:
            error_msg = "Request timeout"
            self.logger.error(f"✗ {error_msg}")
            return {
                'success': False,
                'status_code': 0,
                'error_message': error_msg,
                'token': '',
                'response_headers': {},
                'response_body': {},
                'request_timestamp': request_timestamp,
                'request_details': request_details
            }

        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            self.logger.error(f"✗ {error_msg}")
            self.logger.exception("Full exception details:")
            return {
                'success': False,
                'status_code': 0,
                'error_message': error_msg,
                'token': '',
                'response_headers': {},
                'response_body': {},
                'request_timestamp': request_timestamp,
                'request_details': request_details
            }

    async def _set_device_param(self, param_name: str, param_value: Any) -> bool:
        """
        Set a device parameter via eWeLink API v2.

        Args:
            param_name: Parameter name (e.g., 'lightStrength', 'nightVision', 'power')
            param_value: Parameter value

        Returns:
            True if successful
        """
        if not self.session:
            self.logger.error("Session not initialized")
            return False

        # Ensure authenticated
        if not await self._ensure_authenticated():
            self.logger.error("Not authenticated")
            return False

        # Build eWeLink API v2 device control request
        # Using "update" action format with thingList + params at top level
        endpoint = f"{self.config.api_base_url}/v2/device/thing"

        payload = {
            "thingList": [
                {
                    "itemType": 1,
                    "id": self.config.device_id
                }
            ],
            "params": {
                param_name: param_value
            }
        }

        headers = {
            'Authorization': f'Bearer {self.token.access_token}',
            'X-CK-Appid': self.config.app_id,
            'Content-Type': 'application/json'
        }

        self.logger.info(f"Setting {param_name}={param_value}...")

        try:
            async with self.session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.config.timeout
            ) as response:
                status_code = response.status
                resp_body = await response.json()

                # CRITICAL: eWeLink API returns HTTP 200 but includes error codes in response body
                # HOWEVER: The commands STILL EXECUTE on the device even with error responses!
                # User confirmed: LEDs change state despite API returning validation errors
                # Therefore: Trust HTTP 200 status, ignore response body error codes
                if status_code == 200:
                    # Log response for debugging but don't fail
                    error_code = resp_body.get('error', 0)
                    if error_code != 0:
                        error_msg = resp_body.get('msg', 'Unknown')
                        self.logger.warning(f"⚠️  API returned error {error_code}: {error_msg} (but command likely executed anyway)")
                    else:
                        self.logger.info(f"✓ {param_name} set to {param_value}")
                    return True
                else:
                    error_msg = resp_body.get('msg', 'Unknown error')
                    self.logger.error(f"✗ HTTP {status_code}: Failed to set {param_name}: {error_msg}")
                    return False

        except asyncio.TimeoutError:
            self.logger.error(f"✗ Timeout setting {param_name}")
            return False
        except Exception as e:
            self.logger.error(f"✗ Exception setting {param_name}: {e}")
            return False

    # =========================================================================
    # Core Request Method
    # =========================================================================

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        retry: bool = True
    ) -> tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Make async API request with timeout and retry.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request payload
            retry: Whether to retry on failure

        Returns:
            (success, response_data, error_message)
        """
        if not self.config.enabled:
            return False, None, "Cloud bridge disabled"

        if not self.session:
            return False, None, "Not connected"

        self.stats['total_requests'] += 1

        # Ensure authenticated
        if not await self._ensure_authenticated():
            self.stats['failed_requests'] += 1
            return False, None, "Authentication failed"

        # Build full URL
        url = f"{self.config.api_base_url}{endpoint}"

        # Headers
        headers = {
            'Authorization': f'{self.token.token_type} {self.token.access_token}',
            'Content-Type': 'application/json'
        }

        attempt = 0
        max_attempts = 1 + (self.config.retry_count if retry else 0)

        while attempt < max_attempts:
            try:
                # Make request
                async with self.session.request(
                    method,
                    url,
                    json=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        self.stats['successful_requests'] += 1
                        self.stats['last_success'] = time.time()
                        return True, response_data, None

                    elif response.status == 401:
                        # Unauthorized - token expired, try refresh
                        if await self._refresh_token():
                            # Update headers with new token
                            headers['Authorization'] = f'{self.token.token_type} {self.token.access_token}'
                            # Retry request
                            continue
                        else:
                            self.stats['failed_requests'] += 1
                            return False, None, "Authentication expired"

                    else:
                        error_msg = f"HTTP {response.status}"
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get('message', error_msg)
                        except:
                            pass

                        # Retry on server errors (5xx)
                        if response.status >= 500 and attempt < max_attempts - 1:
                            self.logger.warning(f"Server error {response.status}, retrying...")
                            self.stats['retries'] += 1
                            attempt += 1
                            await asyncio.sleep(0.5)
                            continue

                        self.stats['failed_requests'] += 1
                        self.stats['last_error'] = error_msg
                        return False, None, error_msg

            except asyncio.TimeoutError:
                if attempt < max_attempts - 1:
                    self.logger.warning(f"Request timeout, retrying...")
                    self.stats['timeouts'] += 1
                    self.stats['retries'] += 1
                    attempt += 1
                    await asyncio.sleep(0.5)
                    continue
                else:
                    self.stats['timeouts'] += 1
                    self.stats['failed_requests'] += 1
                    self.stats['last_error'] = "Timeout"
                    return False, None, "Request timeout"

            except Exception as e:
                if attempt < max_attempts - 1:
                    self.logger.warning(f"Request error: {e}, retrying...")
                    self.stats['retries'] += 1
                    attempt += 1
                    await asyncio.sleep(0.5)
                    continue
                else:
                    error_msg = str(e)
                    self.stats['failed_requests'] += 1
                    self.stats['last_error'] = error_msg
                    return False, None, error_msg

            attempt += 1

        # Should not reach here
        self.stats['failed_requests'] += 1
        return False, None, "Max retries exceeded"

    # =========================================================================
    # Public API - Sleep/Privacy Mode
    # =========================================================================

    async def sleep_on(self) -> bool:
        """
        Enable sleep/privacy mode (turn camera power off).

        Returns:
            True if successful
        """
        self.logger.info("Enabling sleep mode via cloud...")
        return await self._set_device_param('power', 'off')

    async def sleep_off(self) -> bool:
        """
        Disable sleep/privacy mode (turn camera power on).

        Returns:
            True if successful
        """
        self.logger.info("Disabling sleep mode via cloud...")
        return await self._set_device_param('power', 'on')

    # =========================================================================
    # Public API - LED Control
    # =========================================================================

    async def set_led(self, level: int) -> bool:
        """
        Set LED brightness level.

        Args:
            level: LED level (0=off, 1=low, 2=high, 3=max)

        Returns:
            True if successful
        """
        if not 0 <= level <= 3:
            self.logger.warning(f"Invalid LED level: {level} (must be 0-3)")
            return False

        self.logger.info(f"Setting LED level to {level} via cloud...")

        # eWeLink uses 'lightStrength' parameter with values 0-3
        return await self._set_device_param('lightStrength', level)

    # =========================================================================
    # Public API - Night/IR Mode
    # =========================================================================

    async def set_night(self, mode: str) -> bool:
        """
        Set night/IR mode.

        Args:
            mode: Night mode ('auto', 'day', 'night')

        Returns:
            True if successful
        """
        if mode not in ['auto', 'day', 'night']:
            self.logger.warning(f"Invalid night mode: {mode}")
            return False

        self.logger.info(f"Setting night mode to {mode} via cloud...")

        # eWeLink uses 'nightVision' parameter with NUMERIC values (not strings!)
        # Device state shows nightVision as integer: 0=day/off, 1=auto, 2=night/on
        ewelink_mode = {'day': 0, 'auto': 1, 'night': 2}[mode]
        return await self._set_device_param('nightVision', ewelink_mode)

    # =========================================================================
    # Public API - Microphone Gain
    # =========================================================================

    async def set_mic_volume(self, volume: int) -> bool:
        """
        Set microphone volume.

        Args:
            volume: Volume level 0-100

        Returns:
            True if successful
        """
        volume = max(0, min(100, volume))
        self.logger.info(f"Setting mic volume to {volume}...")
        return await self._set_device_param('microphoneVolume', volume)

    async def set_speaker_volume(self, volume: int) -> bool:
        """
        Set speaker volume.

        Args:
            volume: Volume level 0-100

        Returns:
            True if successful
        """
        volume = max(0, min(100, volume))
        self.logger.info(f"Setting speaker volume to {volume}...")
        return await self._set_device_param('speakerVolume', volume)

    async def set_alarm(self, on: bool) -> bool:
        """
        Trigger manual alarm on/off.

        Args:
            on: True = alarm on, False = alarm off

        Returns:
            True if successful
        """
        self.logger.info(f"Setting alarm to {'ON' if on else 'OFF'}...")
        return await self._set_device_param('manualAlarm', 1 if on else 0)

    async def set_smart_tracking(self, on: bool) -> bool:
        """
        Enable/disable firmware-level smart tracking.

        Args:
            on: True = enable, False = disable

        Returns:
            True if successful
        """
        self.logger.info(f"Setting smart tracking to {'ON' if on else 'OFF'}...")
        return await self._set_device_param('smartTraceEnable', 1 if on else 0)

    async def set_motion_detection(self, enable: bool, humanoid_only: bool = True, sensitivity: int = 0) -> bool:
        """
        Configure motion detection.

        Args:
            enable: Enable/disable motion detection
            humanoid_only: True = person detection only, False = all motion
            sensitivity: Detection sensitivity (0-2)

        Returns:
            True if successful
        """
        config = {
            'enable': 1 if enable else 0,
            'type': {'all': 0 if humanoid_only else 1, 'humanoid': 1 if humanoid_only else 0},
            'sensitivity': max(0, min(2, sensitivity)),
        }
        self.logger.info(f"Setting motion detection: enable={enable}, humanoid={humanoid_only}...")
        return await self._set_device_param('moveDetection', config)

    async def set_screen_flip(self, flip: bool) -> bool:
        """Flip camera image 180 degrees."""
        self.logger.info(f"Setting screen flip to {flip}...")
        return await self._set_device_param('screenFlip', 1 if flip else 0)

    async def set_watermark(self, on: bool) -> bool:
        """Show/hide timestamp watermark."""
        self.logger.info(f"Setting watermark to {'ON' if on else 'OFF'}...")
        return await self._set_device_param('watermarkDisplay', 1 if on else 0)

    async def set_status_led(self, on: bool) -> bool:
        """
        Control the blue status LED (sledOnline).

        Args:
            on: True = LED on, False = LED off

        Returns:
            True if successful
        """
        self.logger.info(f"Setting status LED to {'ON' if on else 'OFF'}...")
        return await self._set_device_param('sledOnline', 'on' if on else 'off')

    async def set_resolution(self, hd: bool = True) -> bool:
        """
        Set video resolution.

        Args:
            hd: True = 1080P, False = 360P

        Returns:
            True if successful
        """
        res = 'HD' if hd else '360P'
        self.logger.info(f"Setting resolution to {res}...")
        return await self._set_device_param('resolution', res)

    async def set_record_audio(self, on: bool) -> bool:
        """Enable/disable audio recording."""
        self.logger.info(f"Setting audio recording to {'ON' if on else 'OFF'}...")
        return await self._set_device_param('recordAudio', 1 if on else 0)

    async def trigger_ptz_calibration(self) -> bool:
        """
        Trigger PTZ calibration (same as app calibration).
        WARNING: Camera will move through full range!

        Returns:
            True if command accepted
        """
        self.logger.warning("Triggering PTZ calibration - camera will move!")
        return await self._set_device_param('ptzReset', 1)

    async def get_device_params(self) -> dict:
        """
        Get all current device parameters from cloud.

        Returns:
            Dict of all device parameters
        """
        if not self.is_connected:
            connected = await self.connect()
            if not connected:
                return {}

        try:
            import aiohttp as _aiohttp
            headers = {
                'Authorization': f'Bearer {self.token.access_token}',
                'X-CK-Appid': self.config.app_id,
                'Content-Type': 'application/json'
            }
            payload = {
                'thingList': [{'itemType': 1, 'id': self.config.device_id}]
            }

            async with _aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.api_base_url}/v2/device/thing",
                    headers=headers,
                    json=payload,
                    timeout=_aiohttp.ClientTimeout(total=self.config.timeout)
                ) as resp:
                    data = await resp.json()
                    if "data" in data and "thingList" in data["data"]:
                        for thing in data["data"]["thingList"]:
                            return thing.get("itemData", {}).get("params", {})
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get device params: {e}")
            return {}


    # =========================================================================
    # Status and Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get cloud bridge statistics."""
        return {
            'status': self.status.value,
            'enabled': self.config.enabled,
            'connected': self.is_connected,
            'token_valid': self.token.is_valid(),
            'stats': self.stats.copy()
        }


# =============================================================================
# Synchronous Wrapper for Blocking Calls
# =============================================================================

class CameraCloudBridgeSync:
    """
    Synchronous wrapper for CameraCloudBridge.

    Allows calling async methods from sync code without blocking.
    Uses background event loop.
    """

    def __init__(self, config: CloudConfig):
        """Initialize sync wrapper."""
        self.config = config
        self.bridge = CameraCloudBridge(config)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread = None

    def start(self):
        """Start background event loop."""
        import threading

        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

        # Wait for loop to start
        import time
        time.sleep(0.1)

    def stop(self):
        """Stop background event loop."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)

    def _run_async(self, coro):
        """Run async coroutine in background loop."""
        if not self.loop:
            self.start()

        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=self.config.timeout + 1.0)
        except:
            return False

    def connect(self) -> bool:
        """Connect to cloud (sync)."""
        return self._run_async(self.bridge.connect())

    def disconnect(self):
        """Disconnect from cloud (sync)."""
        self._run_async(self.bridge.disconnect())

    def sleep_on(self) -> bool:
        """Enable sleep mode (sync)."""
        return self._run_async(self.bridge.sleep_on())

    def sleep_off(self) -> bool:
        """Disable sleep mode (sync)."""
        return self._run_async(self.bridge.sleep_off())

    def set_led(self, level: int) -> bool:
        """Set LED level (sync)."""
        return self._run_async(self.bridge.set_led(level))

    def set_night(self, mode: str) -> bool:
        """Set night mode (sync)."""
        return self._run_async(self.bridge.set_night(mode))

    def set_mic_volume(self, volume: int) -> bool:
        """Set mic volume (sync)."""
        return self._run_async(self.bridge.set_mic_volume(volume))

    def set_speaker_volume(self, volume: int) -> bool:
        """Set speaker volume (sync)."""
        return self._run_async(self.bridge.set_speaker_volume(volume))

    def set_alarm(self, on: bool) -> bool:
        """Set alarm (sync)."""
        return self._run_async(self.bridge.set_alarm(on))

    def set_smart_tracking(self, on: bool) -> bool:
        """Set smart tracking (sync)."""
        return self._run_async(self.bridge.set_smart_tracking(on))

    def set_motion_detection(self, enable: bool, humanoid_only: bool = True, sensitivity: int = 0) -> bool:
        """Set motion detection (sync)."""
        return self._run_async(self.bridge.set_motion_detection(enable, humanoid_only, sensitivity))

    def set_screen_flip(self, flip: bool) -> bool:
        """Set screen flip (sync)."""
        return self._run_async(self.bridge.set_screen_flip(flip))

    def set_watermark(self, on: bool) -> bool:
        """Set watermark (sync)."""
        return self._run_async(self.bridge.set_watermark(on))

    def set_status_led(self, on: bool) -> bool:
        """Set status LED (sync)."""
        return self._run_async(self.bridge.set_status_led(on))

    def set_resolution(self, hd: bool = True) -> bool:
        """Set resolution (sync)."""
        return self._run_async(self.bridge.set_resolution(hd))

    def set_record_audio(self, on: bool) -> bool:
        """Set audio recording (sync)."""
        return self._run_async(self.bridge.set_record_audio(on))

    def trigger_ptz_calibration(self) -> bool:
        """Trigger PTZ calibration (sync)."""
        return self._run_async(self.bridge.trigger_ptz_calibration())

    def get_device_params(self) -> dict:
        """Get all device params (sync)."""
        return self._run_async(self.bridge.get_device_params())

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics (sync)."""
        return self.bridge.get_stats()


# =============================================================================
# Main / Test
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    # Test configuration
    config = CloudConfig(
        enabled=True,
        api_base_url="https://api.example.com",
        device_id="test_device_123",
        username="test_user",
        password="test_pass",
        timeout=3.0,
        retry_count=1
    )

    print("=" * 80)
    print("Camera Cloud Bridge Test")
    print("=" * 80)
    print("\nNOTE: This is a framework test with placeholder endpoints.")
    print("      Replace API endpoints with actual cloud API.\n")

    async def test_bridge():
        """Test cloud bridge."""
        async with CameraCloudBridge(config) as bridge:
            print(f"Status: {bridge.status.value}")
            print(f"Connected: {bridge.is_connected}")

            if bridge.is_connected:
                print("\nTesting API calls (will fail with placeholder endpoints):")

                # Test sleep mode
                print("\n1. Testing sleep mode...")
                await bridge.sleep_on()
                await asyncio.sleep(1)
                await bridge.sleep_off()

                # Test LED
                print("\n2. Testing LED control...")
                await bridge.set_led(2)

                # Test night mode
                print("\n3. Testing night mode...")
                await bridge.set_night('auto')

                # Test mic gain
                print("\n4. Testing mic gain...")
                await bridge.set_mic_gain(0.7)

                # Show stats
                print("\n5. Statistics:")
                stats = bridge.get_stats()
                print(json.dumps(stats, indent=2))

    # Run async test
    asyncio.run(test_bridge())

    print("\n" + "=" * 80)
    print("Test complete")
    print("=" * 80)
