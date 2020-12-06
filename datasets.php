<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Datasets - LHSPred</title>
        <link rel = "stylesheet" type = "text/css" href = "css/main.css" />
    </head>
    <body>
        <div class = "section_header">
            <center><p class="title">LHSPred - Lung Health Severity Prediction</p></center>
        </div>

        <div class = "section_menu">
            <center>
            <table cellpadding="3px">
                <tr class="nav">
                    <td class="nav"><a href="index.html" class="side_nav">Home</a></td>
                    <td class="nav"><a href="about.html" class="side_nav">About</a></td>
                    <td class="nav"><a href="help.html" class="side_nav">Help</a></td>
                    <td class="nav"><a href="#" class="active">Datasets</a></td>
                    <td class="nav"><a href="team.html" class="side_nav">Team</a></td>
                    <td class="nav"><a href="https://github.com/ttsudipto/lhspred" class="side_nav">Source (GitHub)</a></td>
                </tr>
            </table>
            </center>
        </div>

        <div class = "section_left">
        <br/>
            <?php
                if($_GET["type"] === "tt") {
                    $csvFileName = "public/tt.csv.txt";
                    $heading = "Training dataset";
                    $tRef = "\"#\"";
                    $tClass = "\"active\"";
                    $vRef = "\"datasets.php?type=val\"";
                    $vClass = "\"side_nav\"";
                }
                elseif ($_GET["type"] === "val") {
                    $csvFileName = "public/val.csv.txt";
                    $heading = "Validation dataset";
                    $tRef = "\"datasets.php?type=tt\"";
                    $tClass = "\"side_nav\"";
                    $vRef = "\"#\"";
                    $vClass = "\"active\"";
                }
                echo "<div class = \"side_nav\"> <a href = ".$tRef." class = ".$tClass."> Training dataset </a> </div>";
                echo "<div class = \"side_nav\"> <a href = ".$vRef." class = ".$vClass."> Validation dataset </a> </div>";
            ?>
        </div>
        
        <div class = "section_right">
            <?php
                $text1 = "<p style=\"font-size: 1.2em; width:76%; margin: 0% 12% 0% 12%; \">
                            The 
                            <a href=\"https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-18786-x/MediaObjects/41467_2020_18786_MOESM4_ESM.xlsx\">
                            original dataset</a> was published by Feng Z. <i>et al.</i>
                            (<a href=\"https://doi.org/10.1038/s41467-020-18786-x\">https://doi.org/10.1038/s41467-020-18786-x</a>).
                            </p><br/>";
        
                $csvFile = fopen($csvFileName, "r");
                
                echo "<center><h1>".$heading."</h1></center>";
                echo $text1;
                echo "<p style=\"font-size: 1.2em; width:76%; margin: 0% 12% 0% 12%; \">
                        To download this data, <a href=\"".$csvFileName."\">Click here</a>.</p><br/>";
                echo "<table class = \"form\" border = \"1\" cellpadding=\"3px\" width=\"100%\">";
                $rowNum = 0;
                while (($row = fgetcsv($csvFile)) !== FALSE) {
                    echo "<tr>";
                    for($i = 0; $i < count($row); ++$i) {
                        if($rowNum == 0)
                            echo "<th>".$row[$i]."</th>";
                        else
                            echo "<td>".$row[$i]."</td>";
                    }
                    echo "</tr>";
                    $rowNum++;
                }
                echo "</table>";
            ?>
            <br/><hr/>
            <p>&nbsp;</p>
        </div>
    </body>
</html>
