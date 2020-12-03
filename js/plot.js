function get_density(div_id, score)
{
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            plot_density(div_id, this.responseText, score);
        }
    };
    xmlhttp.open('GET', 'output/densities/densities.csv', true);
    xmlhttp.setRequestHeader("Content-type", "text/csv");
    xmlhttp.send();
}

function get_column(csv, index)
{
    var column = []
    for(var i=1; i<csv.length; ++i)
    {
        column[i] = csv[i][index];
    }
    return column;
}

function plot_one(div_id, x, pos_y, neg_y, score)
{
    var graphDiv = document.getElementById(div_id);
    var pos_curve = {
        x: x,
        y: pos_y,
        name: 'High risk density',
        line: {width: 3}
    };
    var neg_curve = {
        x: x,
        y: neg_y,
        name: 'Low risk density',
        line: {width: 3}
    };
    var score_pt = {
        mode: 'markers',
        x: [score],
        y: [0],
        name: 'Predicted CT severity score',
        marker: {
            color: 'red',
            size: 12
        }
    }
    var data = [neg_curve, pos_curve, score_pt];
    var layout = {
        title : {
            text: 'Density plot and the predicted CT severity score',
            font: {size: 24}
        },
        xaxis: {
            visible : true,
            color: 'black',
            width: 2,
            tickfont: {size: 14},
            title : {
                text : 'CT severity scores',
                font: {size: 18}
            }
        },
        yaxis: {
            visible : true,
            color: 'black',
            width: 2,
            ticks: 'outside',
            ticklen: 5,
            tickwidth: 1,
            tickfont: {size: 14},
            title : {
                text : 'Probability',
                font: {size: 18}
            }
        },
        shapes: [
            {
                type: 'line',
                yref: 'paper',
                x0: score,
                y0: 0,
                x1: score,
                y1: 1,
                name: 'Threshold',
                line:{
                    dash:'dot',
                    color: 'black',
                    width:1
                }
            }
        ],
        legend: {
            font: {size: 15}
        },
        plot_bgcolor: '#ffe6cc',//'#e6e6e6',
        paper_bgcolor: '#ffe6cc',//'#e6e6e6',
//         margin: {t:0}
    };

    Plotly.plot(graphDiv, data, layout, {showSendToCloud:true});
}

function plot_density(div_id, response, score)
{
    var csv = $.csv.toArrays(response);
    
    var x = get_column(csv, 0);
    var pos_y = get_column(csv, 2);
//     var neg_x = get_column(csv, 2);
    var neg_y = get_column(csv, 1);
    
    plot_one(div_id, x, pos_y, neg_y, score);
}
