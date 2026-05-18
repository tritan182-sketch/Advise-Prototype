# voice_bridge.ps1
Add-Type -AssemblyName System.Speech
$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$recognizer.SetInputToDefaultAudioDevice()

# Build your exact voice grammar vocabulary
$choices = New-Object System.Speech.Recognition.Choices
$choices.Add(@("advise scan", "advise snapshot", "advise status", "advise list", "advise export"))

$gb = New-Object System.Speech.Recognition.GrammarBuilder
$gb.Append($choices)
$grammar = New-Object System.Speech.Recognition.Grammar($gb)
$recognizer.LoadGrammar($grammar)

Write-Host "=== HALO WINDOWS VOICE BRIDGE RUNNING ===" -ForegroundColor Cyan
Write-Host "Listening for vocabulary phrases..." -ForegroundColor Yellow

while ($true) {
    # Blocks here offline until your microphone registers a match
    $result = $recognizer.Recognize()
    if ($result -ne $null) {
        Write-Host "Voice Match Detected: '$($result.Text)'" -ForegroundColor Green
        # Drop a small transaction file onto the disk for Python to consume
        $result.Text | Out-File -FilePath "voice_command.tmp" -Encoding utf8
    }
    Start-Sleep -Milliseconds 100
}