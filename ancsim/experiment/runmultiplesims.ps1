
$p0name = "MCPOINTS"
$p0 = 10000
$p1name = "LEARNINGFACTOR"
$p1 = @(1, 0.5)
$p2name = "SPEAKERDIM"
$p2 = @(2, 3, 4, 5)
$p3name = "NOISEFREQ"
$p3 = @(100,200, 300, 400, 500)

#$p3name = "NOISEFREQ"
#$p3 = @(200,210)
#$p3 = 1e-7, 5e-7
#$p3 = @(@(1.5,1.5),@(2.5,2.5), @(3.5,3.5), @(4.5,4.5), @(5.5,5.5))


$date = Get-Date -Format "yyyy/MM/dd"
$time = Get-Date -Format "HH:mm"
$timestr = $date.replace("/", "_") + "_" + $time.replace(":","_")
$foldername = "test_plots_" + $timestr


if (Test-Path (".\"+$foldername+"/"))
{
	$k = 1
	$foldername = $foldername + "_" + $k
	while (Test-Path (".\"+$foldername+"/"))
	{
		$k = $k + 1
		$foldername = $foldername.Substring(0,$foldername.Length-1)
		$foldername = $foldername + $k
	}
}
$foldername = "multi_experiments/" + $foldername + "/"
New-Item -Name $foldername -ItemType "directory"

$descriptorpath = ".\"+$foldername + "test_description.txt"

Add-Content -Path $descriptorpath -Value  ($p0name + " = " +$p0)
Add-Content -Path $descriptorpath -Value  ($p1name + " = " +$p1)
Add-Content -Path $descriptorpath -Value  ($p2name + " = " +$p2)
Add-Content -Path $descriptorpath -Value  ($p3name + " = " +$p3)

foreach ($param0 in $p0)
{
	foreach ($param1 in $p1)
	{
		foreach ($param2 in $p2)
		{
			foreach ($param3 in $p3)
			{
				python setsettings.py $p0name $param0 $p1name $param1 $p2name $param2 $p3name $param3
				python simulator.py $foldername
			}
		}
	}
}



