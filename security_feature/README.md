# 🔒 Security Monitor - Face-based Screen Lock

A Python-based security system that monitors your webcam for authorized faces and automatically locks your screen when unauthorized access is detected.

## 🚀 Features

- **Real-time Face Recognition** - Uses MediaPipe for accurate face detection
- **Liveness Detection** - Prevents spoofing with photo/video attacks
- **Automatic Screen Lock** - Locks screen when unauthorized access detected
- **Auto-Login** - Automatically unlocks computer when your face is detected
- **Easy Registration** - Register faces through web interface
- **Low Resource Usage** - Optimized for continuous monitoring

## 📋 Prerequisites

- Python 3.8+
- Webcam
- Windows 10/11 (for screen lock functionality)

## 🛠️ Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Additional Tools (Windows):**
   - Download `nircmd.exe` from https://www.nirsoft.net/utils/nircmd.html
   - Place `nircmd.exe` in your system PATH or in the security_feature folder

## 🎯 Usage

### Step 1: Register Your Face

1. **Start the Web Interface:**
   ```bash
   cd ../python-backend
   python main.py
   ```

2. **Open Browser:**
   - Go to `http://localhost:8000`
   - Click "Start Camera" and "Connect to Backend"
   - Click "Register Face" and enter your name
   - Look at the camera for 3 seconds to register

### Step 2: Start Security Monitoring

1. **Run Security Monitor:**
   ```bash
   python security_monitor.py
   ```
   
   Or double-click `start_security.bat`

2. **Monitor Status:**
   - Green "AUTHORIZED" = You're detected and verified
   - Red "UNAUTHORIZED" = No face or unauthorized face detected
   - Screen will automatically lock after 3 seconds of unauthorized access
   - If auto-login is enabled, screen will unlock automatically when you return

### Step 3: Enable Auto-Login (Optional)

To automatically unlock your computer when your face is detected:

1. **Setup Auto-Login:**
   ```bash
   python setup_auto_login.py
   ```
   Or double-click `setup_auto_login.bat`

2. **Enter Your Password:**
   - Enter your Windows password when prompted
   - Password will be encrypted and stored securely
   - Only works with the same user account

3. **Test Auto-Login:**
   ```bash
   python test_auto_login.py
   ```

4. **How It Works:**
   - When your face is detected, the system types your password
   - Computer unlocks automatically without manual input
   - Password is encrypted with a unique key for security

### Step 4: Auto-Start with Restart Protection

To start automatically on Windows boot with automatic restart when terminated:

#### Method 1: Robust Restart (Recommended)
1. **Use the robust startup script:**
   - Double-click `startup_security_robust.bat`
   - This uses PowerShell for reliable restart functionality

2. **Add to Startup:**
   - Right-click `startup_security.vbs` → "Create shortcut"
   - Press `Win + R` → type `shell:startup` → Enter
   - Copy the shortcut to the startup folder

#### Method 2: Windows Service (Most Robust)
1. **Install as Service:**
   - Right-click `install_service.bat` → "Run as administrator"
   - This installs the monitor as a Windows Service

2. **Manage Service:**
   - Open Services.msc to start/stop the service
   - Service will automatically start on boot

#### Method 3: Test Restart Functionality
1. **Test the restart:**
   - Run `test_restart.bat`
   - Open Task Manager and end the Python process
   - The monitor should restart automatically within 5 seconds

## ⚙️ Configuration

Edit `security_monitor.py` to adjust settings:

```python
self.lock_delay = 10       # Seconds before locking when face disappears
self.unlock_delay = 1      # Seconds before unlocking when face appears
self.similarity_threshold = 0.90  # Face recognition threshold (0.0-1.0)
```

## 🎮 Controls

- **'q'** - Quit monitoring
- **'r'** - Reload face database
- **'a'** - Toggle auto-login (if configured)
- **Ctrl+C** - Emergency stop

## 🔧 Troubleshooting

### "No registered faces found"
- Make sure you've registered your face using the web interface first
- Check that `face_database.pkl` exists in the python-backend folder

### "Failed to read from webcam"
- Make sure your webcam is not being used by another application
- Try restarting the security monitor

### Screen lock not working
- Make sure `nircmd.exe` is installed and accessible
- Try running as administrator
- Check Windows permissions for screen lock

### Low recognition accuracy
- Register your face in good lighting conditions
- Look directly at the camera during registration
- Adjust `similarity_threshold` in the code

### Application doesn't restart after Task Manager termination
- Use `startup_security_robust.bat` instead of the basic batch file
- Make sure PowerShell execution policy allows scripts: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- For maximum reliability, install as Windows Service using `install_service.bat`
- Check that the VBS file points to the robust batch file

### Auto-login not working
- Make sure you've run `setup_auto_login.py` and entered your password correctly
- Check that `encrypted_password.dat` and `encryption_key.key` files exist
- Try running `test_auto_login.py` to diagnose issues
- Make sure pyautogui and pygetwindow are installed: `pip install pyautogui pygetwindow`
- Auto-login may not work on some Windows versions due to security restrictions

## 🛡️ Security Notes

- **Privacy**: All face data is stored locally on your computer
- **No Internet**: The system works completely offline
- **Encrypted Storage**: Face embeddings are stored securely
- **Liveness Protection**: Prevents photo/video spoofing attacks

## 📊 Performance

- **CPU Usage**: ~15-25% on modern processors
- **Memory**: ~200-400MB RAM
- **Latency**: ~100-200ms detection time
- **Battery**: Minimal impact on laptop battery life

## 🔄 Updates

To update the system:
1. Stop the security monitor
2. Update the code
3. Restart the monitor

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Make sure your webcam is working
4. Check Windows permissions

---

**⚠️ Important**: This is a security tool. Make sure to test it thoroughly before relying on it for sensitive work. Always have a backup method to unlock your computer!
