document.addEventListener("DOMContentLoaded", function() {

  // Generate an array of x values between -1 and 1
  function generateXValues(n = 100) {
    const xValues = [];
    for (let i = 0; i < n; i++) {
      let x = -1 + (2 * i) / (n - 1);
      xValues.push(x);
    }
    return xValues;
  }

  function computeBeta(xValues, b) {
    return xValues.map(x => {
      const base = 1 - x * x;
      const y = Math.pow(base, 4 * Math.exp(b));
      return x * x > 1 ? null : y;
    });
  }

  function computeO(o, N) {
    return 1 - Math.pow(1 - o, 1 / N);
  }

  function computeDensification(yValues, o, N) {
    return yValues.map(x => {
      return 1 - Math.pow(1 - computeO(o, N) * x, N);
    });
  }

  // Generate x data for the plot
  const xValues = generateXValues();

  // Get slider elements
  const b_slider = document.getElementById('b-mcmc');
  const o_slider = document.getElementById('o-mcmc');
  const N_slider = document.getElementById('N-mcmc');

  // Function to update the plot data
  function updatePlot() {
    const b = parseFloat(b_slider.value);
    const o = parseFloat(o_slider.value);
    const N = parseFloat(N_slider.value);

    // Compute beta values and multiply each by o
    const betaValues = computeBeta(xValues, b);
    // Compute densification values using the scaled beta values
    const densifiedValues = computeDensification(betaValues, o, N);

    // Update both traces in the plot:
    Plotly.update('plotmcmc', 
      { 
        y: [betaValues.map(val => o * val), densifiedValues] 
      },
      { title: `2D Beta Kernel` },
      [0, 1]
    );
  }

  // Initial computation
  const initialB = parseFloat(b_slider.value);
  const initialO = parseFloat(o_slider.value);
  const initialN = parseFloat(N_slider.value);
  const initialBeta = computeBeta(xValues, initialB);
  const initialDensification = computeDensification(initialBeta, initialO, initialN);

  const trace1 = {
    x: xValues,
    y: initialBeta.map(val => initialO * val),
    mode: 'lines',
    line: { color: 'blue' },
    name: 'Beta'
  };

  const trace2 = {
    x: xValues,
    y: initialDensification,
    mode: 'lines',
    line: { color: 'red' },
    name: 'Densification'
  };

  const layout = {
    title: `2D Beta Kernel`,
    xaxis: { title: 'x', range: [-1.1, 1.1] },
    yaxis: { title: 'f(x)', range: [0, 1.1] },
    legend: {
      orientation: 'h',
      x: 0.5,
      y: 1,
      xanchor: 'center',
      yanchor: 'bottom',
      bgcolor: 'rgba(255,255,255,0)'  // transparent background
    },
    margin: { t: 0, b: 0, l: 10, r: 10 }
  };

  // Render the initial plot in the element with id "plotmcmc"
  Plotly.newPlot('plotmcmc', [trace1, trace2], layout);

  // Add event listeners for each slider to update the plot when values change
  b_slider.addEventListener('input', updatePlot);
  o_slider.addEventListener('input', updatePlot);
  N_slider.addEventListener('input', updatePlot);
});
