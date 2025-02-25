// Wrap all code in DOMContentLoaded to ensure the DOM is fully loaded before execution.
document.addEventListener("DOMContentLoaded", function() {

    // Generate an array of x values between -1 and 1
    function generateXValues(n = 500) {
      const xValues = [];
      for (let i = 0; i < n; i++) {
        let x = -1 + (2 * i) / (n - 1);
        xValues.push(x);
      }
      return xValues;
    }

    // Compute y values for f(x) = (1-x^2)^b
    function computeYValues(xValues, b) {
      return xValues.map(x => {
        const base = 1 - x * x;
        // Use Math.pow; if the result is NaN (i.e. for negative base with non-integer exponent), use null.
        const y = Math.pow(base, 4 * Math.exp(b));
        return isNaN(y) ? null : y;
      });
    }

    // Compute y values for the Gaussian
    function computeGaussian(xValues) {
        return xValues.map(x => Math.exp(-9 * x * x / 2));
      }

    // Generate x data for the plot
    const xValues = generateXValues();

    // Get initial b value from the slider
    const slider = document.getElementById('interpolation-slider');
    let b = parseFloat(slider.value);
    let yValues = computeYValues(xValues, b);
    let gaussianValues = computeGaussian(xValues);

    const trace1 = {
        x: xValues,
        y: yValues,
        mode: 'lines',
        line: { color: 'blue' },
        name: `Beta`
      };

    const trace2 = {
    x: xValues,
    y: gaussianValues,
    mode: 'lines',
    line: { color: 'red' },
    name: 'Gaussian'
    };

    const layout = {
      title: ``,
      xaxis: { title: 'x', range: [-1.1, 1.1] },
      yaxis: { title: 'f(x)' },
      legend: {
        orientation: 'h',
        x: 0.5,
        y: 1,
        xanchor: 'center',
        yanchor: 'bottom',
        bgcolor: 'rgba(255,255,255,0)'  // transparent background
      },
      margin: { t: 40, b: 50, l: 50, r: 50 }
    };

    // Render the initial plot in the element with id "plot"
    Plotly.newPlot('plot', [trace1, trace2], layout);

    // Update the plot when the slider value changes
    slider.addEventListener('input', function() {
      b = parseFloat(this.value);
      yValues = computeYValues(xValues, b);
      Plotly.update('plot', { y: [yValues] }, { title: `` }, [0]);
    });
  });