// Wrap all code in DOMContentLoaded to ensure the DOM is fully loaded before execution.
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
          if (Array.isArray(x)) {
            // Recursively process each row if x is an array (2D case)
            return computeBeta(x, b);
          } else {
            const base = 1 - x * x;
            const y = Math.pow(base, 4 * Math.exp(b));
            return x*x>1 ? null : y;
          }
        });
      }

    function computeGaussian(xValues) {
        return xValues.map(x => {
            if (Array.isArray(x)) {
                // Recursively process each row if x is an array (2D case)
                return computeGaussian(x, b);
            } else {
                return Math.exp(-9 * x * x / 2);
            }
        });
      }

    // Generate x data for the plot
    const xValues = generateXValues();

    // Get initial b value from the slider
    const slider = document.getElementById('interpolation-slider');
    let b = parseFloat(slider.value);
    let yValues = computeBeta(xValues, b);
    const gaussianValues = computeGaussian(xValues);

    // Define custom colorscale
    const customColorscale = [
        [0.0, '#264653'],
        [0.25, '#2a9d8f'],
        [0.5, '#e9c46a'],
        [0.75, '#f4a261'],
        [1.0, '#e76f51']
    ];

    const trace1 = {
        x: xValues,
        y: yValues,
        mode: 'lines',
        line: { color: '#264653' },  // Using first color from scale
        name: `Beta`
    };

    const trace2 = {
        x: xValues,
        y: gaussianValues,
        mode: 'lines',
        line: { color: '#e76f51' },  // Using last color from scale
        name: 'Gaussian'
    };

    const layout = {
      title: `2D Beta Kernel`,
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
      margin: { t: 0, b: 0, l: 10, r: 10 }
    };

    // Render the initial plot in the element with id "plot"
    Plotly.newPlot('plot', [trace1, trace2], layout);

    // Generate grid data for the 3D surface
    function generateGrid(n = 100) {
        const xGrid = generateXValues(n);
        const yGrid = generateXValues(n);
        const rGrid = [];
        // Create a 2D array for z values.
        // Each row corresponds to one b value and each column to one x value.
        for (let j = 0; j < n; j++) {
            let row = [];
            let y = yGrid[j];
            for (let i = 0; i < n; i++) {
            let x = xGrid[i];
            const r = Math.sqrt(x * x + y * y);
            // let z = Math.pow(1-r, 4 * Math.exp(0));
            row.push(r);
            }
            rGrid.push(row);
        }
        return { xGrid, yGrid, rGrid };
    }

    const { xGrid, yGrid, rGrid } = generateGrid();
    let zGrid = computeBeta(rGrid, b);

    // Define the 3D surface trace
    const trace3d = {
        x: xGrid,
        y: yGrid,
        z: zGrid,
        type: 'surface',
        colorscale: customColorscale,
        cmin: 0,
        cmax: 1,
        colorbar: { title: 'Beta', tickvals: [0, 0.5, 1] },
        name: 'Beta Surface',
        opacity: 0.99,
        showscale: false,
    };

    const trace3d_gaussian = {
        x: xGrid,
        y: yGrid,
        z: computeGaussian(rGrid),
        type: 'surface',
        colorscale: [
          [0, '#264653'],
          [1, '#e76f51']
        ],
        cmin: 0,
        cmax: 1,
        colorbar: { title: 'Gaussian', tickvals: [0, 0.5, 1] },
        name: 'Gaussian Surface',
        opacity: 0.35,
        showscale: false,
    };

    const layout3d = {
    title: '3D Beta Kernel',
    scene: {
        xaxis: { title: 'x', range: [-1, 1], showgrid: false, showline: false, zeroline: false, showticklabels: false },
        yaxis: { title: 'y', range: [-1, 1], showgrid: false, showline: false, zeroline: false, showticklabels: false },
        zaxis: { title: 'Beta', range: [0, 1], showgrid: false, showline: false, zeroline: false, showticklabels: false }
    },
    margin: { t: 0, b: 0, l: 10, r: 10 },
    };

    // Render the 3D surface plot
    Plotly.newPlot('plot3d', [trace3d, trace3d_gaussian], layout3d);

    // Update the plot when the slider value changes
    slider.addEventListener('input', function() {
        b = parseFloat(this.value);
        yValues = computeBeta(xValues, b);
        Plotly.update('plot', { y: [yValues] }, { title: `2D Beta Kernel` }, [0]);
        zGrid = computeBeta(rGrid, b);
        Plotly.update('plot3d', { z: [zGrid] }, { title: '3D Beta Kernel' }, [0]);
    });
  });