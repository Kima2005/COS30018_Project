var initialDateStr = new Date().toUTCString();

var ctx = document.getElementById("chart").getContext("2d");

var barData = [];
var lineData = [];
let dateArray = []
let predictionLineData = [];
let predictionLineData2 = [];

fetchCSVData("Amazon");

var chart = new Chart(ctx, {
    type: "candlestick",
    data: {
        datasets: [
            {
                label: "Amazon Stock Data",
                data: barData,
                hidden: true,
            },
            {
                label: "Close price",
                type: "line",
                data: lineData,
            },

            // prediction
            {
                label: "Prediction close price",
                type: "line",
                data: predictionLineData,
            },

            // prediction 2
            {
                label: "Prediction close price 2",
                type: "line",
                data: predictionLineData2,
            }
        ],
    },

    options: {
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'day'
                }
            }
        },
        plugins: {
            zoom: {
                zoom: {
                    wheel: {
                        enabled: true,
                        speed: 0.2,
                    },
                    pinch: {
                        enabled: true,
                    },
                    mode: "x",
                    speed: 0.2,
                    limits: {
                        x: {min: 'original', max: 'original', minRange: 1000 * 60 * 60 * 24 * 7} // minRange is set to 7 days
                    },
                    time: {
                        unit: 'day'
                    }
                },
                pan: {
                    enabled: true,
                    mode: "x",
                    threshold: 2,
                    limits: {
                        x: {min: 'original', max: 'original'} // Limits for panning
                    },
                    time: {
                        unit: 'day'
                    }
                },
            },
        },
    },
});


async function fetchCSVData(name) {
    barData = [];
    lineData = [];
    dateArray = [];
    predictionLineData = [];
    predictionLineData2 = [];

    const predict_res = fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ filename: name })
    });

    const predict_res2 = fetch("http://127.0.0.1:8000/predict2", {
        method: "POST",
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ filename: name })
    });

    const response = await fetch(prefixName(name));
    const data = await response.text();
    const parsedData = parseCSV(data);

    parsedData.forEach((row) => {
        if (row.Date === undefined) return;

        const date = luxon.DateTime.fromFormat(
            row.Date,
            "yyyy-MM-dd"
        ).valueOf();

        dateArray.push(date);
        barData.push({
            x: date,
            o: parseFloat(row.Open),
            h: parseFloat(row.High),
            l: parseFloat(row.Low),
            c: parseFloat(row.Close),
        });

        lineData.push({
            x: date,
            y: parseFloat(row.Close),
        });
    });

    chart.config.data.datasets[0].data = barData;
    chart.config.data.datasets[1].data = lineData;
    chart.config.data.datasets[0].label = stockData(name);
    chart.canvas.parentNode.style.height = "80vh";
    chart.canvas.parentNode.style.width = "80vw";

    chart.update();

    await predict_res.then((data) => {
        return data.json();
    }).then((d) => {
        const data = d.predictions;
        for (let index = 0; index < dateArray.length; index++) {
            const prediction = data[index] ?? 0;

            if (prediction !== 0) {
                predictionLineData.push({
                    x: dateArray[index],
                    y: prediction,
                });
            }

            // Set close price to zero if prediction is zero
            if (prediction === 0) {
                lineData[index].y = 0;
            }
        }

        // Filter out zero values from lineData
        lineData = lineData.filter(dataPoint => dataPoint.y !== 0);

        chart.config.data.datasets[1].data = lineData;
        chart.config.data.datasets[2].data = predictionLineData;
        chart.update();
    }).catch(err => console.log(err));

    await predict_res2.then((data) => {
        return data.json();
    }).then((d) => {
        const data = d.predictions;
        console.log(data)
        for (let index = 0; index < dateArray.length; index++) {
            const prediction = data[index] ?? 0;

            if (prediction !== 0) {
                predictionLineData2.push({
                    x: dateArray[index],
                    y: prediction,
                });
            }

            // console.log(lineData)
            // Set close price to zero if prediction is zero
            // if (prediction === 0) {
            //     lineData[index].y = 0;
            // }
        }

        // Filter out zero values from lineData
        lineData = lineData.filter(dataPoint => dataPoint.y !== 0);

        chart.config.data.datasets[1].data = lineData;
        chart.config.data.datasets[3].data = predictionLineData2;
        chart.update();
    }).catch(err => console.log(err));
}



function parseCSV(data) {
    const lines = data.split("\n");
    const headers = lines[0].split(",");
    return lines.slice(1).map((line) => {
        const values = line.split(",");
        let obj = {};
        headers.forEach((header, index) => {
            if (values[index] === undefined) return;
            obj[header.trim()] = values[index].trim();
        });
        return obj;
    });
}

var update = function () {
    var dataset = chart.config.data.datasets[0];

    // candlestick vs ohlc
    var type = document.getElementById("type").value;
    chart.config.type = type;

    // linear vs log
    var scaleType = document.getElementById("scale-type").value;
    chart.config.options.scales.y.type = scaleType;

    // color
    var colorScheme = document.getElementById("color-scheme").value;
    if (colorScheme === "neon") {
        chart.config.data.datasets[0].backgroundColors = {
            up: "#01ff01",
            down: "#fe0000",
            unchanged: "#999",
        };
    } else {
        delete chart.config.data.datasets[0].backgroundColors;
    }

    // border
    var border = document.getElementById("border").value;
    if (border === "false") {
        dataset.borderColors = "rgba(0, 0, 0, 0)";
    } else {
        delete dataset.borderColors;
    }

    // mixed charts
    var mixed = document.getElementById("mixed").value;
    if (mixed === "true") {
        chart.config.data.datasets[1].hidden = false;
    } else {
        chart.config.data.datasets[1].hidden = true;
    }

    chart.update();
};

[...document.getElementsByTagName("select")].forEach((element) =>
    element.addEventListener("change", update)
);

document.getElementById("csv-file").addEventListener("change", function () {
    var selectedFile = this.value;
    fetchCSVData(selectedFile);
});

function prefixName(name) {
    return `./resource/${name}_final.csv`;
}

function stockData(name) {
    return `${name} Stock Data`;
}