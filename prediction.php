<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Home - LHSPred</title>
        <link rel = "stylesheet" type = "text/css" href = "css/main.css" />
        <script type = "text/javascript" src = "js/plot.js"></script>
        <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-csv/1.0.8/jquery.csv.min.js"></script>
        <script type = "text/javascript" src = "https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <style>
            * {
                box-sizing: border-box;
            }
            .plot_container {
                width: 80%;
                margin-left: 10%;
/*                 height: 300px; */
            }
            .plot_container::after {
                content: "";
                clear: both;
/*                 display: table; */
            }
            .plot_div {
                width: 100%;
                height: 300px;
            }
            .plot_caption {
                width: 80%;
                margin-left: 10%;
                text-align: center;
                background-color: #cce6ff;
                font-weight: bold;
            }
            th {
                border: 1px black solid;
            }
        </style>
        <div class = "section_header">
            <center><p class="title">LHSPred - Lung Health Severity Prediction</p></center>
        </div>

        <div class = "section_menu">
            <center>
            <table cellpadding="3px">
                <tr class="nav">
                    <td class="nav"><a href="index.html" class="active">Home</a></td>
                    <td class="nav"><a href="about.html" class="side_nav">About</a></td>
                    <td class="nav"><a href="help.html" class="side_nav">Help</a></td>
                    <td class="nav"><a href="datasets.php?type=tt" class="side_nav">Datasets</a></td>
                    <td class="nav"><a href="team.html" class="side_nav">Team</a></td>
                </tr>
            </table>
            </center>
        </div>

        <!--<div class = "section_left"></div>-->
        
        <div class = "section_middle">
            <?php
                include 'validation.php';
                $valid = validateInput();
                if($valid == false) {
                    echo "<center><h3>Error !!</h3>";
//                     echo "<p>".$errMsg."</p>";
                    exit("<i><b>Error Message:</b> ".$errMsg."</i><p>Please try again.</p><a href=\"index.html\"><input type=\"button\" value=\"Check again\" /></a></center>");
                }
                
                $params = array();
                $params["age"] = $_POST["age"];
                $params["wbc"] = $_POST["wbc"];
                $params["nlr"] = $_POST["nlr"];
                $params["ast"] = $_POST["ast"];
                $params["albumin"] = $_POST["albumin"];
                $params["ldh"] = $_POST["ldh"];
                $params["crp"] = $_POST["crp"];
                $params["model_id"] = $_POST["model_id"];
                $arg_json = json_encode($_POST);
//                 echo $arg_json;
                
//                 $command = "venv3.7/bin/python3 -m python.driver '".$arg_json."' 2>&1";
                $command = "venv2.7/bin/python -m python.driver '".$arg_json."' 2>&1";
//                 echo "<pre>".$command."</pre>\n";
                exec($command, $out, $status);
                
                $json_out = "";
//                 for ($i=0;$i<count($out);++$i)
//                     echo $out[$i]."<br/>";
                for ($i=0;$i<count($out);++$i)
                    if(substr($out[$i], 0, 8) === "JSON-OP>") {
                        $json_out = substr($out[$i], 8);
                        break;
                    }
//                 echo $json_out."<br/>";
             
                $result = json_decode($json_out);
//                 print_r($result);
            ?>
            <center><h2>Input</h2></center>
            <table class = "form" border = "1" cellpadding="3px" id = "stable">
                <tr>
                    <td style="padding-left : 5px;">Age (years)</td>
                    <td><center><?php echo $_POST["age"]; ?></center></td>
                </tr>
                <tr>
                    <td style="padding-left : 5px;">White blood cell count (x10<sup>9</sup>/L)</td>
                    <td><center><?php echo $_POST["wbc"]; ?></center></td>
                </tr>
                <tr>
                    <td style="padding-left : 5px;">Neutrophil-to-Lymphocyte ratio</td>
                    <td><center><?php echo $_POST["nlr"]; ?></center></td>
                </tr>
                <tr>
                    <td style="padding-left : 5px;">Aspartate transaminase - AST (U/L)</td>
                    <td><center><?php echo $_POST["ast"]; ?></center></td>
                </tr>
                <tr>
                    <td style="padding-left : 5px;">Albumin (g/L)</td>
                    <td><center><?php echo $_POST["albumin"]; ?></center></td>
                </tr>
                <tr>
                    <td style="padding-left : 5px;">Lactate dehydrogenase - LDH (U/L)</td>
                    <td><center><?php echo $_POST["ldh"]; ?></center></td>
                </tr>
                <tr>
                    <td style="padding-left : 5px;">C-reactive protein (mg/L)</td>
                    <td><center><?php echo $_POST["crp"]; ?></center></td>
                </tr>
                <tr>
                    <th style="border: 1px black solid;">Regression model</th>
                    <td>
                        <center>
                        <?php 
                            if($params["model_id"] == "SVR")
                                echo "Support Vector Regressor (SVR)";
                            elseif ($params["model_id"] == "MLPR")
                                echo "Multi-layer Perceptron Regressor (MLPR)";
                        ?>
                        </center>
                    </td>
                </tr>
            </table><br/>
            <center><a href="index.html"><input type="button" value="Check again" /></a></center>
            <br/><hr/>
            <center><h2>Result</h2></center>
            <table class = "form" border = "1" cellpadding="5px" id = "rtable">
                <tr>
                    <th>Regressor</th>
                    <th>Predicted CT severity score</th>
                    <th>High risk of pneumonia confidence (%)</th>
                    <th>Low risk of pneumonia confidence (%)</th>
                </tr>
                <?php
                    echo "<tr>";
                    echo "<td><center><b>".$params["model_id"]."</b></center></td>";
                    echo "<td><center>".round(floatval($result->score), 3)."</center></td>";
                    echo "<td><center>".round(floatval($result->positiveness), 3)."</center></td>";
                    echo "<td><center>".round(floatval($result->negativeness), 3)."</center></td>";
                    echo "</tr>";
                ?>
            </table>
            <br/><br/>
            <div class="plot_container" id="plot_container"></div>
<!--             <div class="plot_caption">Classifier 0</div><br/> -->
<!--             <div id="plt_div0" class="plot_div"><?php /*echo "<script>get_density('plt_div0',0,".$result->scores[0].",".$result->thresholds[0].");</script>";*/ ?></div> -->
            <?php echo "<script>get_density('plot_container',".round(floatval($result->score), 3).");</script>"; ?>
            
            <br/><hr/>
            <p>&nbsp;</p>
        </div>
    </body>
</html>
