param(
    [string]$TaskName = 'Chatbot-Data-Pipeline',
    [string]$At = '03:00',
    [switch]$Replace
)
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
$runnerPath = Join-Path $projectPath 'run_pipeline.py'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Virtual environment missing: $pythonPath" }
if ((Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) -and -not $Replace) {
    throw 'Task already exists. Use -Replace explicitly to update it.'
}
$arguments = '-X utf8 "' + $runnerPath + '" --mode collect --resume-pending'
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument $arguments -WorkingDirectory $projectPath
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 12) -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description 'Collection -> canonical -> chunks -> incremental embedding -> Qdrant/BM25 sync'
Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force:$Replace | Out-Null
Write-Output "Registered $TaskName daily at $At (Windows local time, while the user is logged in)."
