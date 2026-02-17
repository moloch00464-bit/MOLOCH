#!/bin/bash
cd /home/molochzuhause/moloch

echo "PTZ Simplification"
echo "=================="

cp core/gui/moloch_unified_panel.py /tmp/panel_backup.py

# 1. Remove ST button (lines 579-582)
echo "1. Removing ST button..."
sed -i '579,582d' core/gui/moloch_unified_panel.py

# 2. Change auto_btn to AUTONOM
echo "2. Changing auto_btn..."
sed -i 's/text="MANUELL", bg="#1a1a3e", fg="white", width=9,/text="AUTONOM", bg="#00aa00", fg="white", width=12,/' core/gui/moloch_unified_panel.py

# 3. Remove st_btn.config lines
echo "3. Removing st_btn.config..."
sed -i '/self\.st_btn\.config/d' core/gui/moloch_unified_panel.py

# 4. Remove _toggle_smart_tracking
echo "4. Removing _toggle_smart_tracking..."
python3 << 'EOFPY'
with open('core/gui/moloch_unified_panel.py') as f:
    lines = f.readlines()
new_lines = []
skip = False
for line in lines:
    if 'def _toggle_smart_tracking' in line and line.strip().startswith('def'):
        skip = True
        continue
    if skip and line.strip().startswith('def ') and '_toggle_smart_tracking' not in line:
        skip = False
    if not skip:
        new_lines.append(line)
with open('core/gui/moloch_unified_panel.py', 'w') as f:
    f.writelines(new_lines)
print("  Removed")
EOFPY

# 5. Simplify _toggle_autonomous
echo "5. Simplifying _toggle_autonomous..."
python3 << 'EOFPY'
import re
with open('core/gui/moloch_unified_panel.py') as f:
    code = f.read()

pattern = r'(    def _toggle_autonomous\(self\):.*?\n)(.*?)(\n    def [a-z_])'
replacement = r'''\1        """Toggle AUTONOM/MANUELL."""
        import tkinter as tk
        if not self.service:
            return
        current_auto = getattr(self.service, '_autonomous_mode', True)
        new_auto = not current_auto
        if isinstance(self.service, ServiceProxy):
            self.service._send_cmd({"action": "toggle_autonomous"})
            self.service._autonomous_mode = new_auto
        else:
            if hasattr(self.service, 'toggle_autonomous_manual'):
                self.service.toggle_autonomous_manual()
        if new_auto:
            self.auto_btn.config(text="AUTONOM", bg="#00aa00")
            logger.info("[MODE] AUTONOM")
        else:
            self.auto_btn.config(text="MANUELL", bg="#dd2222")
            logger.info("[MODE] MANUELL")
\3'''
code = re.sub(pattern, replacement, code, flags=re.DOTALL, count=1)
with open('core/gui/moloch_unified_panel.py', 'w') as f:
    f.write(code)
print("  Simplified")
EOFPY

echo ""
echo "Syntax Check..."
python3 -m py_compile core/gui/moloch_unified_panel.py
if [ $? -eq 0 ]; then
    echo "OK"
else
    echo "ERROR - Restoring"
    cp /tmp/panel_backup.py core/gui/moloch_unified_panel.py
    exit 1
fi

echo ""
echo "Service Restart..."
sudo systemctl restart moloch.service
sleep 3

systemctl is-active moloch.service | grep -q active && echo "Service OK" || exit 1

echo ""
echo "DONE - Panel neustarten"
