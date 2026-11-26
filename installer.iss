; Script Inno Setup untuk Halotec Container OCR Desktop Application
; Menghasilkan 1 file setup installer mandiri (Setup_HalotecContainerOCR.exe)

#define MyAppName "Halotec Container OCR - Drone View"
#define MyAppVersion "1.9.0"
#define MyAppPublisher "Halotec"
#define MyAppURL "https://example.com"
#define MyAppExeName "HalotecContainerOCR.exe"

[Setup]
AppId={{D8E5F921-9B5A-4B6A-8A4C-73E92AC5D430}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\Halotec\ContainerOCR
DefaultGroupName=Halotec Container OCR
DisableProgramGroupPage=yes
OutputDir=dist_installer
OutputBaseFilename=Setup_HalotecContainerOCR_v1.9.0
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest

[Languages]
Name: "indonesian"; MessagesFile: "compiler:Languages\Indonesian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Copy all files and DLLs generated from dist/HalotecContainerOCR
Source: "dist\HalotecContainerOCR\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Filename: "{app}\{#MyAppExeName}"; Flags: nowait postinstall skipifsilent
