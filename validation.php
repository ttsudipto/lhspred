<?php
    $errMsg = "";
    function checkExistence() {
        global $errMsg;
        if (!array_key_exists("age", $_POST)) {
            $errMsg = "Age value does not exist";
            return false;
        }
        if (!array_key_exists("wbc", $_POST)) {
            $errMsg = "WBC value does not exist";
            return false;
        }
        if (!array_key_exists("nlr", $_POST)) {
            $errMsg = "NLR value does not exist";
            return false;
        }
        if (!array_key_exists("ast", $_POST)) {
            $errMsg = "AST value does not exist";
            return false;
        }
        if (!array_key_exists("albumin", $_POST)) {
            $errMsg = "Albumin value does not exist";
            return false;
        }
        if (!array_key_exists("ldh", $_POST)) {
            $errMsg = "LDH value does not exist";
            return false;
        }
        if (!array_key_exists("crp", $_POST)) {
            $errMsg = "CRP value does not exist";
            return false;
        }
        if (!array_key_exists("model_id", $_POST)) {
            $errMsg = "Regression model does not exist";
            return false;
        }
        return true;
    }
    
    function checkRanges() {
        global $errMsg;
        if(!is_numeric($_POST["age"]) or intval($_POST["age"])<1) {
            $errMsg = "Invalid Age value";
            return false;
        }
        if(!is_numeric($_POST["wbc"]) or floatval($_POST["wbc"])<0) {
            $errMsg = "Invalid WBC value";
            return false;
        }
        if(!is_numeric($_POST["nlr"]) or floatval($_POST["nlr"])<0) {
            $errMsg = "Invalid NLR value";
            return false;
        }
        if(!is_numeric($_POST["ast"]) or floatval($_POST["ast"])<0) {
            $errMsg = "Invalid AST value";
            return false;
        }
        if(!is_numeric($_POST["albumin"]) or floatval($_POST["albumin"])<0) {
            $errMsg = "Invalid Albumin value";
            return false;
        }
        if(!is_numeric($_POST["ldh"]) or floatval($_POST["ldh"])<0) {
            $errMsg = "Invalid LDH value";
            return false;
        }
        if(!is_numeric($_POST["crp"]) or floatval($_POST["crp"])<0) {
            $errMsg = "Invalid CRP value";
            return false;
        }
//         echo "<p>".$_POST["model_id"]."</p>";
//         echo strcmp($_POST["model_id"], "SVR");
//         echo strcmp($_POST["model_id"], "SVR");
        if(strcmp($_POST["model_id"], "SVR") != 0 and strcmp($_POST["model_id"], "MLPR") != 0) {
            $errMsg = "Invalid model selected";
            return false;
        }
        return true;
    }
    
    function validateInput() {
        if(!checkExistence())
            return false;
        if(!checkRanges())
            return false;
        return true;
    }
?>
