document.addEventListener("DOMContentLoaded", () => {
  // Generate an array of n values between -1 and 1.
  function generateXValues(n) {
    const step = 2 / (n - 1);
    const arr = new Array(n);
    for (let i = 0; i < n; i++) {
      arr[i] = -1 + step * i;
    }
    return arr;
  }

  // Compute Beta values for each x in xValues, using b.
  function computeBeta(xValues, b) {
    const expb = Math.exp(b);
    return xValues.map(x => (x * x > 1 ? null : Math.pow(1 - x * x, 4 * expb)));
  }

  // Compute a scaling factor from o and N.
  function computeO(o, N) {
    return 1 - Math.pow(1 - o, 1 / N);
  }

  // Compute densification values given yValues, o, and N.
  function computeDensification(yValues, o, N) {
    const factor = computeO(o, N);
    return yValues.map(x => 1 - Math.pow(1 - factor * x, N));
  }

  // --- Line Plot (ID: plotmcmc) ---
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
    Plotly.update(
      "plotmcmc",
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
    line: { color: "blue" },
    name: "Beta"
  };
  const trace2 = {
    x: xValues,
    y: initialDensification,
    mode: "lines",
    line: { color: "red" },
    name: "Densification"
  };
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

  // --- Splat Plot (ID: plotmcmc_splat) ---
  // Generates circle data for three circles:
  // • Beta Circle: centered at (0,2), opacity = beta (blue)
  // • Densification Circle: centered at (0,0), opacity = densification (red)
  // • Error Circle: centered at (0,-2), opacity = |beta – densification| (green)
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
          // Shift positions for each circle.
          betaCircle.x.push(xi);
          betaCircle.y.push(yj + 2);
          betaCircle.colors.push(`rgba(0,0,255,${betaVal.toFixed(2)})`);

          densCircle.x.push(xi);
          densCircle.y.push(yj);
          densCircle.colors.push(`rgba(0,255,0,${densVal.toFixed(2)})`);

          errorCircle.x.push(xi);
          errorCircle.y.push(yj - 2);
          errorCircle.colors.push(`rgba(255,0,0,${errorVal.toFixed(2)})`);
        }
      }
    }
    return { betaCircle, densCircle, errorCircle };
  }

  function updateSplatPlot() {
    const b = parseFloat(b_slider.value);
    const o = parseFloat(o_slider.value);
    const N = parseFloat(N_slider.value);
    const { betaCircle, densCircle, errorCircle } = generateCircleData(b, o, N, 50);
    Plotly.update("plotmcmc_splat", {
      x: [betaCircle.x, densCircle.x, errorCircle.x],
      y: [betaCircle.y, densCircle.y, errorCircle.y],
      "marker.color": [betaCircle.colors, densCircle.colors, errorCircle.colors]
    });
  }

  const initialCircleData = generateCircleData(initialB, initialO, initialN, 50);
  const traceBetaSplat = {
    x: initialCircleData.betaCircle.x,
    y: initialCircleData.betaCircle.y,
    mode: "markers",
    marker: { size: 6, color: initialCircleData.betaCircle.colors },
    name: "Beta"
  };
  const traceDensSplat = {
    x: initialCircleData.densCircle.x,
    y: initialCircleData.densCircle.y,
    mode: "markers",
    marker: { size: 6, color: initialCircleData.densCircle.colors },
    name: "Densification"
  };
  const traceErrorSplat = {
    x: initialCircleData.errorCircle.x,
    y: initialCircleData.errorCircle.y,
    mode: "markers",
    marker: { size: 6, color: initialCircleData.errorCircle.colors },
    name: "Error"
  };
  const layoutSplat = {
    title: "2D Beta Splat",
    xaxis: { title: "x", range: [-2, 2] },
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
  Plotly.newPlot("plotmcmc_splat", [traceBetaSplat, traceDensSplat, traceErrorSplat], layoutSplat);

  // Combined update function.
  function updateAll() {
    updatePlot();
    updateSplatPlot();
  }
  b_slider.addEventListener("input", updateAll);
  o_slider.addEventListener("input", updateAll);
  N_slider.addEventListener("input", updateAll);
});
