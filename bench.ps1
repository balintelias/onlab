# Measure-Command {start-process & C:\Users\Elijah\Documents\EGYETEM\6\onlab\onlab\run_sim.ps1 -Wait}

# Start script
$targetScriptPath = "C:\Users\Elijah\Documents\EGYETEM\6\onlab\onlab\run_sim.ps1"

# Measure the time taken to execute the target script
$executionTime = Measure-Command {
    # Execute the target script
    & $targetScriptPath
}

# Output the results
Write-Output "The script $targetScriptPath took $($executionTime.TotalSeconds) seconds to execute."