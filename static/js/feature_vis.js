// ------------ utils ------------
const seg = (s) => encodeURIComponent(s ?? "");

// If there is a mapping for a local path, return the HLS URL; else return the local path.
function mapOrLocal(localPath) {
  const m = (window.VIDEO_MAP || {});
  return m[localPath] || localPath;
}

// Build path per instance (A vs B)
const buildFeatureSrc = (scene, kernel, feature, ext='mp4') => {
  const local = `./static/videos/${seg(scene)}/${seg(kernel)}/${seg(feature)}.${ext}`;
  return mapOrLocal(local);
};

// Attention maps live in an extra folder:
const buildAttentionSrc = (scene, kernel, feature, ext='mp4') => {
  const local = `./static/videos/${seg(scene)}/${seg(kernel)}/attention_map/${seg(feature)}.${ext}`;
  return mapOrLocal(local);
};

// Quantized features live in an extra folder:
const buildQuantizedSrc = (scene, kernel, feature, ext='mp4') => {
  const local = `./static/videos/${seg(scene)}/${seg(kernel)}/quantized/${seg(feature)}.${ext}`;
  return mapOrLocal(local);
};

// Segmentation results live in an extra folder:
const buildSegmentationSrc = (scene, kernel, feature, ext='mp4') => {
  const local = `./static/videos/${seg(scene)}/${seg(kernel)}/segmentation/${seg(feature)}.${ext}`;
  return mapOrLocal(local);
};

function canPlay(ext) {
  const v = document.createElement('video');
  const mime = ext === 'webm'
    ? 'video/webm; codecs="vp9,opus"'
    : 'video/mp4; codecs="avc1.42E01E,mp4a.40.2"';
  return v.canPlayType(mime) !== '';
}

// Label/value options per scene ONLY for the Attention Query window (B)
const QUERIES_BY_SCENE_B = {
  Figurines: [
    { value: "bag_of_snacks_with_pikachu_on_it", label: "pikachu" },
    { value: "blue_elephant",                    label: "blue elephant" },
    { value: "green_apple",                      label: "green apple" },
    { value: "green_chair",                      label: "green chair" },
    { value: "jake_the_dog",                     label: "jake" },
    { value: "red_apple",                        label: "red apple" },
    { value: "red_box",                          label: "red box" },
    { value: "rubber_yellow_duck",               label: "rubber yellow duck" },
    { value: "waldo",                            label: "waldo" },
  ],
  Ramen: [
    // TODO: replace with your real ramen queries & filenames
    { value: "chopsticks",    label: "chopsticks" },
    { value: "glass_cup",     label: "glass cup" },
    { value: "half_of_egg",   label: "half of egg" },
    { value: "iron_sake_cup", label: "sake cup" },
    { value: "kamaboko",      label: "kamaboko" },
    { value: "nori",          label: "nori" },
    { value: "silver_plate",  label: "plate" },
    { value: "table",         label: "table" },
    { value: "weavy_noodle",  label: "weavy noodle" },
    { value: "white_napkin",  label: "white napkin" },
  ],
};


// Label/value options per scene ONLY for the Attention Query window (B)
const QUERIES_BY_SCENE_D = {
  Figurines: [
    { value: "bag_of_snacks_with_pikachu_on_it", label: "pikachu" },
    { value: "blue_elephant",                    label: "blue elephant" },
    { value: "green_apple",                      label: "green apple" },
    { value: "green_chair",                      label: "green chair" },
    { value: "jake_the_dog",                     label: "jake" },
    { value: "red_apple",                        label: "red apple" },
    { value: "red_box",                          label: "red box" },
    { value: "rubber_yellow_duck",               label: "rubber yellow duck" },
    { value: "waldo",                            label: "waldo" },
    { value: "RGB",                              label: "RGB" },
  ],
  Ramen: [
    // TODO: replace with your real ramen queries & filenames
    { value: "chopsticks",    label: "chopsticks" },
    { value: "glass_cup",     label: "glass cup" },
    { value: "half_of_egg",   label: "half of egg" },
    { value: "iron_sake_cup", label: "sake cup" },
    { value: "kamaboko",      label: "kamaboko" },
    { value: "nori",          label: "nori" },
    { value: "silver_plate",  label: "plate" },
    { value: "table",         label: "table" },
    { value: "weavy_noodle",  label: "weavy noodle" },
    { value: "white_napkin",  label: "white napkin" },
    { value: "RGB",          label: "RGB" },
  ],
};

// ------------ three-way overlapped compare (scoped per container) ------------
function createThreeWayCompare({
  
  boxSel, tabsSel, buildSrc, queriesByScene=null,
  initialScene="Figurines",
  initialPanes=[ // A, B, C defaults
    { kernel: "3DGS", feature: "maskclip" },
    { kernel: "2DGS", feature: "vit" },
    { kernel: "DBS",  feature: "resnet50" },
  ],
  extOrder=['mp4','webm'],
}) {
  const isHls = (u) => /\.m3u8(\?|$)/.test(u);
  const canNativeHls = () => {
    const v = document.createElement('video');
    return v.canPlayType('application/vnd.apple.mpegURL') !== '';
  };
  const box = document.querySelector(boxSel);
  const tabs = document.querySelectorAll(`${tabsSel} .nav-link`);
  if (!box) return;

  // videos and UI inside THIS box only
  const vids = [...box.querySelectorAll('.layer-video')];
  const kernelSelects  = [...box.querySelectorAll('.kernel-select')];
  const featureSelects = [...box.querySelectorAll('.feature-select')];
  const dividers = [...box.querySelectorAll('.divider')]; // use data-divider

  const state = {
    scene: initialScene,
    edges: [33.33, 66.66], // %
    panes: JSON.parse(JSON.stringify(initialPanes)),
    playing: false,
  };

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function applyClips() {
    const [x0, x1] = state.edges;
    vids[0].style.clipPath = `inset(0 ${100 - x0}% 0 0)`;
    vids[1].style.clipPath = `inset(0 ${100 - x1}% 0 ${x0}%)`;
    vids[2].style.clipPath = `inset(0 0 0 ${x1}%)`;
    const d0 = dividers.find(d=>d.dataset.divider==='0');
    const d1 = dividers.find(d=>d.dataset.divider==='1');
    if (d0) d0.style.left = `calc(${x0}% - 3px)`;
    if (d1) d1.style.left = `calc(${x1}% - 3px)`;
  }
  const isM3U8 = (u) => /\.m3u8(\?|$)/i.test(u);
  const isMPD  = (u) => /\.mpd(\?|$)/i.test(u);
  const isAbs  = (u) => /^https?:\/\//i.test(u);
  
  
  function setVideoSource(i, src, restart=true, autoplay=true, onErr=null) {
    const oldV = vids[i];
  
    // clean up previous adaptive players
    if (oldV._hls) { try { oldV._hls.destroy(); } catch(e){} delete oldV._hls; }
    if (oldV._dash){ try { oldV._dash.reset();   } catch(e){} delete oldV._dash; }
  
    const nv = oldV.cloneNode(true);
    nv.muted = true; 
    nv.playsInline = true;
  
    if (onErr) {
      nv.addEventListener('error', () => {
        console.error(`[${boxSel}] pane ${i} failed:`, src, nv.error);
        onErr();
      }, { once: true });
    }
  
    nv.addEventListener('loadedmetadata', () => {
      if (restart) nv.currentTime = 0;
      if (autoplay) nv.play().catch(()=>{});
    }, { once: true });
  
    let adaptive = false;
  
    if (isM3U8(src)) {
      if (nv.canPlayType('application/vnd.apple.mpegurl')) {
        nv.src = src; nv.load();
        adaptive = true;
      } else if (window.Hls && window.Hls.isSupported()) {
        const hls = new Hls();
        hls.on(Hls.Events.ERROR, (_, data) => {
          if (data?.fatal && onErr) onErr();
        });
        hls.loadSource(src);
        hls.attachMedia(nv);
        nv._hls = hls;
        adaptive = true;
      }
    } else if (isMPD(src) && window.dashjs) {
      const p = dashjs.MediaPlayer().create();
      p.initialize(nv, src, true);
      nv._dash = p;
      adaptive = true;
    }
  
    if (!adaptive) { nv.src = src; nv.load(); }
  
    oldV.parentNode.replaceChild(nv, oldV);
    vids[i] = nv;
    applyClips();
  }
  

  function updatePaneSrc(i, {restart=true, autoplay=true} = {}) {
    const { kernel, feature } = state.panes[i];
    // Filter ext order by browser support
    const playable = extOrder.filter(canPlay);
    const tryExts = (playable.length ? playable : extOrder).slice();
    let k = 0;
    const tryLoad = () => {
      const ext = tryExts[k];
      const src = buildSrc(state.scene, kernel, feature, ext);
      // console.log(`[${boxSel}] pane ${i} -> ${src}`);
      setVideoSource(i, src, restart, autoplay, () => {
        if (++k < tryExts.length) tryLoad();
      });
    };
    tryLoad();
  }

  function updateAllSrc(opts={restart:true, autoplay:true}) {
    for (let i=0;i<3;i++) updatePaneSrc(i, opts);
  }

  function restartAllFromStartAndPlay() {
    state.playing = true;
    vids.forEach(v => { try { v.currentTime = 0; v.play(); } catch(e){} });
  }

  // click anywhere in box to toggle play/pause (ignore controls)
  box.addEventListener('click', (e) => {
    if (e.target.tagName === "SELECT" || e.target.closest(".controls")) return;
    state.playing = !state.playing;
    vids.forEach(v => state.playing ? v.play().catch(()=>{}) : v.pause());
  });

  // Dragging logic (two dividers)
  const MIN_GAP = 10; // %
  let dragging = null; // 0|1
  function onMove(clientX) {
    if (dragging === null) return;
    const r = box.getBoundingClientRect();
    let pct = ((clientX - r.left) / r.width) * 100;
    pct = clamp(pct, 0, 100);
    let [x0, x1] = state.edges;
    if (dragging === 0) {
      x0 = clamp(pct, MIN_GAP, x1 - MIN_GAP);
    } else {
      x1 = clamp(pct, x0 + MIN_GAP, 100 - MIN_GAP);
    }
    state.edges = [x0, x1];
    applyClips();
  }
  dividers.forEach(div => {
    div.addEventListener('mousedown', (e)=>{ dragging = Number(div.dataset.divider); document.body.style.cursor="col-resize"; e.preventDefault(); });
    div.addEventListener('touchstart', (e)=>{ dragging = Number(div.dataset.divider); document.body.style.cursor="col-resize"; }, {passive:false});
  });
  window.addEventListener('mousemove', (e)=>onMove(e.clientX));
  window.addEventListener('touchmove', (e)=>onMove(e.touches[0].clientX), {passive:false});
  window.addEventListener('mouseup', ()=>{ dragging=null; document.body.style.cursor=""; });
  window.addEventListener('touchend', ()=>{ dragging=null; document.body.style.cursor=""; });

  // Populate feature selects for a scene (only if queriesByScene is provided)
  function populateFeatureSelectsForScene(scene) {
    if (!queriesByScene) return; // A window: static list from HTML
    const options = queriesByScene[scene] || [];
    featureSelects.forEach(sel => {
      const prev = sel.value;
      sel.innerHTML = "";
      for (const {value,label} of options) {
        const o = document.createElement('option');
        o.value = value; o.textContent = label;
        sel.appendChild(o);
      }
      // keep previous if still present, else first
      const keep = options.some(o => o.value === prev);
      sel.value = keep ? prev : (options[0]?.value ?? "");
      const i = Number(sel.dataset.vid);
      state.panes[i].feature = sel.value;
    });
  }

  kernelSelects.forEach(sel => {
    const i = Number(sel.dataset.vid);
    // force UI to reflect initialPanes
    sel.value = state.panes[i].kernel ?? sel.value;
    sel.addEventListener('change', () => {
      state.panes[i].kernel = sel.value;
      updatePaneSrc(i, {restart:true, autoplay:true});
      restartAllFromStartAndPlay();
    });
  });
  
  // features (for A the HTML list is static; for D/B we also repopulate per scene)
  featureSelects.forEach(sel => {
    const i = Number(sel.dataset.vid);
    // force UI to reflect initialPanes
    sel.value = state.panes[i].feature ?? sel.value;
    sel.addEventListener('change', () => {
      state.panes[i].feature = sel.value;
      updatePaneSrc(i, {restart:true, autoplay:true});
      restartAllFromStartAndPlay();
    });
  });

  // Tab click → set scene and (if applicable) repopulate queries
  tabs.forEach((a, idx) => {
    a.addEventListener('click', () => {
      tabs.forEach(x => x.classList.remove('active'));
      a.classList.add('active');
      state.scene = (idx === 1) ? 'Ramen' : 'Figurines';
      populateFeatureSelectsForScene(state.scene);
      updateAllSrc({restart:true, autoplay:true});
      restartAllFromStartAndPlay();
    });
  });

  // init
  applyClips();
  populateFeatureSelectsForScene(state.scene);
  updateAllSrc({restart:true, autoplay:false}); // load; user click toggles play

  return {
    setScene(scene){ state.scene = scene; populateFeatureSelectsForScene(scene); updateAllSrc({restart:true, autoplay:true}); },
    restart(){ restartAllFromStartAndPlay(); },
  };
}

// ------------ boot both sections ------------
document.addEventListener('DOMContentLoaded', () => {
  // Feature Visualization (A): static feature list (from HTML), simple path
  createThreeWayCompare({
    boxSel:  '#geom-compare-A',
    tabsSel: '#geometry-decomposition-A',
    buildSrc: buildFeatureSrc,
    initialScene: 'Figurines',
    // extOrder: ['mp4','webm'], // default
  });

  // Attention Query (B): queries depend on scene, path includes attention_map
  createThreeWayCompare({
    boxSel:  '#geom-compare-B',
    tabsSel: '#geometry-decomposition-B',
    buildSrc: buildAttentionSrc,
    queriesByScene: QUERIES_BY_SCENE_B,
    initialScene: 'Figurines',
    // Prefer MP4 first unless you specifically want WebM first:
    // extOrder: ['webm','mp4'],
  });

  // Quantized Features (C): queries depend on scene, path includes quantized
  createThreeWayCompare({
    boxSel:  '#geom-compare-C',
    tabsSel: '#geometry-decomposition-C',
    buildSrc: buildQuantizedSrc,
    queriesByScene: QUERIES_BY_SCENE_B,
    initialScene: 'Figurines',
    // Prefer MP4 first unless you specifically want WebM first:
    // extOrder: ['webm','mp4'],
  });

  // Quantized Features (D):segementation results
  createThreeWayCompare({
    boxSel:  '#geom-compare-D',
    tabsSel: '#geometry-decomposition-D',
    buildSrc: buildSegmentationSrc,
    queriesByScene: QUERIES_BY_SCENE_D,
    initialScene: 'Figurines',
    initialPanes: [
      { kernel: "3DGS", feature: "rubber_yellow_duck" },
      { kernel: "2DGS", feature: "RGB" },
      { kernel: "DBS",  feature: "waldo" },
    ],
    // Prefer MP4 first unless you specifically want WebM first:
    // extOrder: ['webm','mp4'],
  });
});
