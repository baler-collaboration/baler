## Windows 10/11 Setup
* Install "git for windows": https://github.com/git-for-windows/git/releases/tag/v2.39.1.windows.1
  * For a 64 bit system, probably use this one: https://github.com/git-for-windows/git/releases/download/v2.39.1.windows.1/Git-2.39.1-64-bit.exe
* Go to your windows search bar and search for "powershell". right-click powerhsell and select "run as administrator"
* Enable Linux subsystem by entering this into the PowerShell and hitting enter: `Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux`
* Go to the windows store and download "Ubuntu 22.04.1 LTS"
* Once downloaded, open it. This will start Ubuntu as a "terminal". After picking a username and password, input the following commands into that terminal. You can copy the comands using ctrl+c or the button to the right of the text. But pasting it into the terminal can only be done by right-clicking anywhere in the terminal window.

Start by updating the Windows Subsystem for Linux
```console
wsl.exe --update
```
Then, synch your clock:
```console
sudo hwclock --hctosys
```
Update your Linux packages
```console
sudo apt-get update
```
Configure git to use tour windows credentials helper, this is necessary for you to authenticate yourself on GitHub.
```console
git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/bin/git-credential-manager-core.exe"
```
Install pip3 for downloading python packages
```console
sudo apt-get install python3-pip
```
At this point, you have a working Linux environment and you can follow the Linux/Mac setup in the main readme
