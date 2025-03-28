document.addEventListener("DOMContentLoaded", () => {
  
  // Define custom colorscale
  const customColorscale = [
    [0.0, '#264653'],
    [0.25, '#2a9d8f'],
    [0.5, '#e9c46a'],
    [0.75, '#f4a261'],
    [1.0, '#e76f51']
  ];

  // Generate an array of n values between -1 and 1.
  function generateXValues(n) {
    const step = 2 / (n - 1);
    const arr = new Array(n);
    for (let i = 0; i < n; i++) {
      arr[i] = -1 + step * i;
    }
    return arr;
  }

  // Compute Beta values for each x in xValues, using parameter b.
  function computeBeta(xValues, b) {
    const expb = Math.exp(b);
    return xValues.map(x => (x * x > 1 ? null : Math.pow(1 - x * x, 4 * expb)));
  }

  // Compute the densification factor.
  function computeO(o, N) {
    return 1 - Math.pow(1 - o, 1 / N);
  }
  // Compute densification values from beta values.
  function computeDensification(yValues, o, N) {
    const factor = computeO(o, N);
    return yValues.map(x => 1 - Math.pow(1 - factor * x, N));
  }

  // --- LINE PLOT (ID: plotmcmc) ---
  const n = 100;
  const xValues = generateXValues(n);
  const b_slider = document.getElementById("b-mcmc");
  const o_slider = document.getElementById("o-mcmc");
  const N_slider = document.getElementById("N-mcmc");

  function updatePlot() {
    const b = parseFloat(b_slider.value);
    const o = parseFloat(o_slider.value);
    const N = parseFloat(N_slider.value);
    const beta = computeBeta(xValues, b);
    const scaledBeta = beta.map(val => o * val);
    const densified = computeDensification(beta, o, N);
    Plotly.update("plotmcmc", 
      { y: [scaledBeta, densified] },
      { title: "2D Beta Kernel" },
      [0, 1]
    );
  }

  const initialB = parseFloat(b_slider.value);
  const initialO = parseFloat(o_slider.value);
  const initialN = parseFloat(N_slider.value);
  const initialBeta = computeBeta(xValues, initialB);
  const initialDensification = computeDensification(initialBeta, initialO, initialN);

  const trace1 = {
    x: xValues,
    y: initialBeta.map(val => initialO * val),
    mode: "lines",
    line: { color: "#264653" },
    name: "Beta"
  };
  const trace2 = {
    x: xValues,
    y: initialDensification,
    mode: "lines",
    line: { color: "#e9c46a" },
    name: "Densification"
  };

  // const trace3 = {
  //   x: xValues,
  //   y: initialBeta.map((val, i) => Math.abs((initialO * val) - initialDensification[i])),
  //   mode: "lines",
  //   line: { color: "#e76f51" },
  //   name: "Error"
  // };

  const layoutLine = {
    title: "2D Beta Kernel",
    xaxis: { title: "x", range: [-1.1, 1.1] },
    yaxis: { title: "f(x)", range: [0, 1.1] },
    legend: {
      orientation: "h",
      x: 0.5,
      y: 1,
      xanchor: "center",
      yanchor: "bottom",
      bgcolor: "rgba(255,255,255,0)"
    },
    margin: { t: 0, b: 0, l: 10, r: 10 }
  };
  Plotly.newPlot("plotmcmc", [trace1, trace2], layoutLine);

  // --- SPLAT PLOT (ID: plotmcmc_splat) ---
  // This function generates data for three circles:
  // • Beta Circle: centered at (0,2), opacity = beta (blue)
  // • Densification Circle: centered at (0,0), opacity = densification (red)
  // • Error Circle: centered at (0,-2), opacity = |beta - densification| (green)
  function generateCircleData(b, o, N, res = 50) {
    const xVals = generateXValues(res);
    const expb = Math.exp(b);
    const factor = computeO(o, N);
    const betaCircle = { x: [], y: [], colors: [] };
    const densCircle = { x: [], y: [], colors: [] };
    const errorCircle = { x: [], y: [], colors: [] };

    for (let i = 0; i < res; i++) {
      const xi = xVals[i];
      for (let j = 0; j < res; j++) {
        const yj = xVals[j];
        if (xi * xi + yj * yj <= 1) {
          const r2 = xi * xi + yj * yj;
          const baseVal = Math.pow(1 - r2, 4 * expb);
          const betaVal = baseVal * o;
          const densVal = 1 - Math.pow(1 - factor * baseVal, N);
          const errorVal = Math.abs(betaVal - densVal);
          // Beta circle at (0,2): shift y upward by 2.
          betaCircle.x.push(xi);
          betaCircle.y.push(yj + 2);
          betaCircle.colors.push(`rgba(38,70,83,${betaVal.toFixed(2)})`);
          // Densification circle at (0,0): no shift.
          densCircle.x.push(xi);
          densCircle.y.push(yj);
          densCircle.colors.push(`rgba(233,196,106,${densVal.toFixed(2)})`);
          // Error circle at (0,-2): shift y downward by 2.
          errorCircle.x.push(xi);
          errorCircle.y.push(yj - 2);
          errorCircle.colors.push(`rgba(231,111,81,${errorVal.toFixed(2)})`);
        }
      }
    }
    return { betaCircle, densCircle, errorCircle };
  }

  // Create initial splat data.
  let circleData = generateCircleData(initialB, initialO, initialN, 50);

  // Main (visible) splat traces with array colors.
  const traceBetaSplat = {
    x: circleData.betaCircle.x,
    y: circleData.betaCircle.y,
    mode: "markers",
    marker: { size: 6, color: circleData.betaCircle.colors },
    name: "Beta",
    showlegend: false
  };
  const traceDensSplat = {
    x: circleData.densCircle.x,
    y: circleData.densCircle.y,
    mode: "markers",
    marker: { size: 6, color: circleData.densCircle.colors },
    name: "Densification",
    showlegend: false
  };
  const traceErrorSplat = {
    x: circleData.errorCircle.x,
    y: circleData.errorCircle.y,
    mode: "markers",
    marker: { size: 6, color: circleData.errorCircle.colors },
    name: "Error",
    showlegend: false
  };

  // Dummy traces to force legend colors (one constant color each).
  const dummyBeta = {
    x: [null],
    y: [null],
    mode: "markers",
    marker: { size: 6, color: "#264653" },
    name: "Beta",
    legendgroup: "beta",
    showlegend: true,
    visible: "legendonly"
  };
  const dummyDens = {
    x: [null],
    y: [null],
    mode: "markers",
    marker: { size: 6, color: "#e9c46a" },
    name: "Densification",
    legendgroup: "dens",
    showlegend: true,
    visible: "legendonly"
  };
  const dummyError = {
    x: [null],
    y: [null],
    mode: "markers",
    marker: { size: 6, color: "#e76f51" },
    name: "Error",
    legendgroup: "error",
    showlegend: true,
    visible: "legendonly"
  };

  const layoutSplat = {
    title: "2D Beta Splat",
    xaxis: { title: "x", range: [-3, 3] },
    yaxis: { title: "y", range: [-3, 3] },
    legend: {
      orientation: "h",
      x: 0.5,
      y: 1,
      xanchor: "center",
      yanchor: "bottom",
      bgcolor: "rgba(255,255,255,0)"
    },
    margin: { t: 0, b: 0, l: 10, r: 10 }
  };

  // Plot six traces: three visible splat traces and three dummy legend traces.
  Plotly.newPlot("plotmcmc_splat", [
    traceBetaSplat, traceDensSplat, traceErrorSplat,
    dummyBeta, dummyDens, dummyError
  ], layoutSplat);

  // Update the splat plot (both visible and dummy traces).
  function updateSplatPlot() {
    const b = parseFloat(b_slider.value);
    const o = parseFloat(o_slider.value);
    const N = parseFloat(N_slider.value);
    circleData = generateCircleData(b, o, N, 50);
    Plotly.update("plotmcmc_splat", {
      x: [
        circleData.betaCircle.x,
        circleData.densCircle.x,
        circleData.errorCircle.x
      ],
      y: [
        circleData.betaCircle.y,
        circleData.densCircle.y,
        circleData.errorCircle.y
      ],
      "marker.color": [
        circleData.betaCircle.colors,
        circleData.densCircle.colors,
        circleData.errorCircle.colors
      ]
    }, {}, [0, 1, 2]);
    // Update dummy traces with the first color from each set.
    Plotly.update("plotmcmc_splat", {
      "marker.color": [
        ["#264653"],
        ["#e9c46a"],
        ["#e76f51"]
      ]
    }, {}, [3, 4, 5]);
  }

  function updateAll() {
    updatePlot();
    updateSplatPlot();
  }
  b_slider.addEventListener("input", updateAll);
  o_slider.addEventListener("input", updateAll);
  N_slider.addEventListener("input", updateAll);
});
