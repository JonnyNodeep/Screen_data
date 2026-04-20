<#
    Подключение к VPS по SSH. Пароли в файлы не кладём — используйте ключ или ввод вручную.

    Примеры:
      .\scripts\connect-server.ps1 -RemoteHost "203.0.113.50"
      $env:DSDATA_SSH_HOST = "203.0.113.50"; .\scripts\connect-server.ps1

    Вход по приватному ключу (рекомендуется):
      .\scripts\connect-server.ps1 -RemoteHost "203.0.113.50" -IdentityFile "$env:USERPROFILE\.ssh\id_ed25519"
      $env:DSDATA_SSH_KEY = "$env:USERPROFILE\.ssh\id_ed25519"; .\scripts\connect-server.ps1 -RemoteHost "203.0.113.50"

    Если настроен ~/.ssh/config (см. scripts/ssh-config.example):
      $env:DSDATA_SSH_CONFIG_HOST = "dsdata-vps"; .\scripts\connect-server.ps1
      .\scripts\connect-server.ps1 -ConfigHost dsdata-vps

    Переменные окружения (опционально):
      DSDATA_SSH_HOST, DSDATA_SSH_USER (по умолчанию root), DSDATA_SSH_PORT (22),
      DSDATA_SSH_KEY (путь к приватному ключу), DSDATA_SSH_CONFIG_HOST (имя Host из config)
#>
param(
    [string] $RemoteHost = $env:DSDATA_SSH_HOST,
    [string] $User = $(if ($env:DSDATA_SSH_USER) { $env:DSDATA_SSH_USER } else { "root" }),
    [string] $Port = $(if ($env:DSDATA_SSH_PORT) { $env:DSDATA_SSH_PORT } else { "22" }),
    [string] $IdentityFile = $env:DSDATA_SSH_KEY,
    [string] $ConfigHost = $env:DSDATA_SSH_CONFIG_HOST
)

$ErrorActionPreference = "Stop"

if ($ConfigHost) {
    ssh $ConfigHost
    exit $LASTEXITCODE
}

if (-not $RemoteHost) {
    Write-Host @"
Укажите сервер одним из способов:
  .\scripts\connect-server.ps1 -RemoteHost YOUR_IP
  `$env:DSDATA_SSH_HOST = 'YOUR_IP'; .\scripts\connect-server.ps1
Или настройте SSH config и запустите с -ConfigHost dsdata-vps (см. scripts\ssh-config.example).
"@ -ForegroundColor Yellow
    exit 1
}

$sshArgs = [System.Collections.Generic.List[string]]::new()
if ($IdentityFile) {
    $sshArgs.Add("-i")
    $sshArgs.Add($IdentityFile)
}
$sshArgs.Add("-p")
$sshArgs.Add("$Port")
$sshArgs.Add("${User}@${RemoteHost}")

ssh @sshArgs
exit $LASTEXITCODE
