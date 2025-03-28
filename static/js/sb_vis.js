document.addEventListener("DOMContentLoaded", function() {
    // Define custom colorscale
    const customColorscale = [
      [0.0, '#264653'],
      [0.25, '#2a9d8f'],
      [0.5, '#e9c46a'],
      [0.75, '#f4a261'],
      [1.0, '#e76f51']
  ];

    // Helper: create a linearly spaced array.
    function linspace(start, stop, num) {
      let arr = [];
      let step = (stop - start) / (num - 1);
      for (let i = 0; i < num; i++) {
        arr.push(start + step * i);
      }
      return arr;
    }
  
    // Grid resolution
    const n = 100;
    // θ ∈ [0,π] and φ ∈ [0,2π]
    const thetaVals = linspace(0, Math.PI, n);
    const phiVals = linspace(0, 2 * Math.PI, n);
  
    // Precompute the sphere coordinates (X, Y, Z) as 2D arrays.
    let X = [], Y = [], Z = [];
    for (let i = 0; i < n; i++) {
      let rowX = [], rowY = [], rowZ = [];
      let theta = thetaVals[i];
      for (let j = 0; j < n; j++) {
        let phi = phiVals[j];
        rowX.push(Math.sin(theta) * Math.cos(phi));
        rowY.push(Math.sin(theta) * Math.sin(phi));
        rowZ.push(Math.cos(theta));
      }
      X.push(rowX);
      Y.push(rowY);
      Z.push(rowZ);
    }
  
    function computeSphericalBeta(thetaVals, phiVals, b, theta0, phi0) {
      let F = [];
      for (let i = 0; i < thetaVals.length; i++) {
        let row = [];
        let theta = thetaVals[i];
        for (let j = 0; j < phiVals.length; j++) {
          let phi = phiVals[j];
          // Dot product between unit vectors in directions (θ,φ) and (θ₀, φ₀):
          let dot = Math.cos(theta) * Math.cos(theta0) + Math.sin(theta) * Math.sin(theta0) * Math.cos(phi - phi0);
          // Subtract 1 so that when the directions match (dot = 1), the exponent is zero and f = 1.
          let f = dot < 0 ? 0 : Math.pow(dot, 4 * Math.exp(b));
          row.push(f);
        }
        F.push(row);
      }
      return F;
    }
  
    // Get slider elements
    const lambdaSlider = document.getElementById("b-slider");
    const thetaSlider = document.getElementById("theta-slider");
    const phiSlider = document.getElementById("phi-slider");
  
    // Initial slider values.
    let lambdaVal = parseFloat(lambdaSlider.value);
    let theta0Val = parseFloat(thetaSlider.value);
    let phi0Val = parseFloat(phiSlider.value);
  
    // Compute initial surfacecolor (F) using the spherical Beta.
    let F = computeSphericalBeta(thetaVals, phiVals, lambdaVal, theta0Val, phi0Val);
  
    // Define the surface trace.
    const traceSB = {
      x: X,
      y: Y,
      z: Z,
      surfacecolor: F,
      type: 'surface',
      colorscale: customColorscale,
      cmin: 0,
      cmax: 1,
      showscale: false,  // Remove the colorbar.
      opacity: 1
    };
  
    const layoutSB = {
      title: 'Spherical Beta',
      scene: {
        xaxis: { title: 'x', range: [-1, 1], showgrid: false, showline: false, zeroline: true, showticklabels: false },
        yaxis: { title: 'y', range: [-1, 1], showgrid: false, showline: false, zeroline: true, showticklabels: false },
        zaxis: { title: 'z', range: [-1, 1], showgrid: false, showline: false, zeroline: true, showticklabels: false }
      },
      margin: { t: 0, b: 0, l: 0, r: 0 }
    };
  
    // Render the 3D spherical Beta surface.
    Plotly.newPlot("plotsb", [traceSB], layoutSB);
  
    // Update function: re-compute F and update the plot.
    function updateSphericalBeta() {
      lambdaVal = parseFloat(lambdaSlider.value);
      theta0Val = parseFloat(thetaSlider.value);
      phi0Val = parseFloat(phiSlider.value);
      let newF = computeSphericalBeta(thetaVals, phiVals, lambdaVal, theta0Val, phi0Val);
      Plotly.update("plotsb", { surfacecolor: [newF] }, { title: 'Spherical Beta' });
    }
  
    // Attach event listeners to all three sliders.
    lambdaSlider.addEventListener("input", updateSphericalBeta);
    thetaSlider.addEventListener("input", updateSphericalBeta);
    phiSlider.addEventListener("input", updateSphericalBeta);
  });