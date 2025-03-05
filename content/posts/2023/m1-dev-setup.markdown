---
title: "How to Set Up a Dev Environment on a MacBook Pro M1 in 2023"
date: "2023-06-09T16:00:00+01:00"
showToc: true
showReadingTime: true
showBreadcrumbs: true
---

I recently had to set up a new M1 MacBook Pro for development, so I wrote down the steps I took along the way.

### Basic security
The section 'The easy stuff' in Ricard Bejarano's ['Hardening macOS' guide](https://www.bejarano.io/hardening-macos/) covers a few security basics for a new macOS machine (most importantly: disk encryption using `FileVault`, automatic security updates, lock screen after inactivity). The last recommendation in the section covers password managers. This is the approach I took:

### Password manager
I use 1Password as my password manager across multiple devices. To set it up on a new macOS machine, download the installer from: `https://1password.com/downloads/mac/`. From here on you can use the setup QR code (from the emergency kit printout page) to get it up and running quickly.

### Time Machine Backups
Make sure to setup backups via Time Machine. :)

### Terminal emulator and shell
Download iTerm2 from: https://iterm2.com/ .
In macOS Ventura `zsh` is the default shell, which we will keep.
Additionally, install `Oh My Zsh` for convenient management of your zsh shell:
```
 sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```


### Text editor: Vim
Vim is already pre-installed, works for me!

### SSH keys
My understanding is that it's considered best practice to generate new SSH keys for a new machine as opposed to copying the  SSH keys from your old machine. This way the private SSH keys don't have to leave the respective machine and are not flying around on old devices and being forgotten about. [To generate a new key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key) follow the prompts and make sure to enter a passphrase:
```shell
ssh-keygen -t ed25519 -C "your_email@example.com"
```
For convenience I also recommend to use the ssh-agent to manage your keys securely so that you don't have to type the passphrase everytime you use the keys: see [Adding your SSH key to the ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#adding-your-ssh-key-to-the-ssh-agent)


### Homebrew
A great package manager for macOS:
 ```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
 ```
 After the installation follow the suggested next steps to add Homebrew to your PATH.
 
### Rosetta 2
 An app that works in the background and enables you to run apps built for Intel Macs  on an Apple silicon Mac (`arm64`). Many apps already support Apple silicon processors, but there's still software out there (especially older versions of apps) that doesn't run natively on Apple silicon. It's also a recommendation to install Rosetta 2 before installing Docker Desktop.
 To install:
 ```
 softwareupdate --install-rosetta
  ```
 
### Note-taking: Obsidian
Great note-taking app for plain text Markdown files:
 ```
brew install --cask obsidian
 ```
 
### Htop
```
brew install htop
```
 
### tmux
 A great terminal multiplexer that let's you run multiple terminal sessions in a single window (as well as detach and re-attach to sessions) :
```
brew install tmux
```
 
### Python Development
#### IDE: VS Code
To install VS Code download it from: `https://code.visualstudio.com/Download` 
After installing VS Code choose the Python extension via Extensions in the sidebar and search for 'Python' in the Marketplace. It will also ask you to install macOS Command Line Developer Tools if you haven't done already.

#### Pyenv
To manage multiple Python versions, install `pyenv`:
```
brew install pyenv
```
[Set up your shell environment for Pyenv](https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv):
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

Install additional dependencies to be able to build Python, for my M1 Macbook Pro this was:
```
brew install xz
```
(If this is not sufficient and you still encounter issues when building your first Python version as described below, check: https://github.com/pyenv/pyenv/wiki#suggested-build-environment )

List all available versions and install for example Python 3.10.11:
```
pyenv install -l
pyenv install 3.10.11
```
 
### Docker Desktop
Make sure you have `Rosetta 2` installed, then download the image from https://docs.docker.com/desktop/install/mac-install/ and run:
```
sudo hdiutil attach Docker.dmg
sudo /Volumes/Docker/Docker.app/Contents/MacOS/install
sudo hdiutil detach /Volumes/Docker
```
(This default installation will perform privileged configurations once during the installation, for details see: https://docs.docker.com/desktop/mac/permission-requirements/#permission-requirements)
