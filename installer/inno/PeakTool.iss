; Inno Setup script for Peak Analysis Tool (LabOne-style)
#define AppName "Peak Analysis Tool"
#define AppVersion "1.0.0"
#define AppPublisher "Your Org"
#define InstallDirName "{pf}\PeakTool"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={#InstallDirName}
DefaultGroupName={#AppName}
OutputDir=.
OutputBaseFilename=PeakTool-Setup
ArchitecturesInstallIn64BitMode=x64
DisableDirPage=no
DisableProgramGroupPage=no

[Files]
Source: "..\..\dist\PeakService.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\launcher\launch_ui.py"; DestDir: "{app}\launcher"; Flags: ignoreversion

[Icons]
Name: "{group}\Peak Analysis Tool"; Filename: "{code:GetPythonExe}"; Parameters: """{app}\launcher\launch_ui.py"""; WorkingDir: "{app}"
Name: "{group}\Start Service (manual)"; Filename: "{app}\PeakService.exe"; WorkingDir: "{app}"

[Run]
Filename: "{code:GetPythonExe}"; Parameters: """{app}\launcher\launch_ui.py"""; Flags: nowait postinstall skipifsilent

[Code]
function GetPythonExe(Param: string): string;
begin
  { Try to use the system python launcher if available }
  Result := 'py.exe';
end


