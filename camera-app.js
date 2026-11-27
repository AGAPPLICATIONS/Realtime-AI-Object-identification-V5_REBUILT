"use strict";
const byId = id => document.getElementById(id);
const video = byId("camera"), photo = byId("photo"), file = byId("file");
const start = byId("start"), status = byId("status");
const frame = document.createElement("canvas");
frame.width = frame.height = 224;
const ctx = frame.getContext("2d", {willReadFrequently: true});
const area = 224 * 224;
let sessionPromise, generation = 0, stream, timer, photoURL, spoken = "";

function loadScript(path) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = path;
    script.onload = resolve;
    script.onerror = () => reject(new Error("Could not load " + path));
    document.head.append(script);
  });
}

async function loadModel() {
  if (!window.ort) await loadScript("./ort.min.js");
  if (typeof imageLabels === "undefined") await loadScript("./labels.js");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = false;
  ort.env.wasm.wasmPaths = { "ort-wasm.wasm": new URL("./ort-wasm.wasm", document.baseURI).href };
  const response = await fetch("./squeezenet.onnx");
  if (!response.ok) throw new Error("squeezenet.onnx returned HTTP " + response.status);
  return ort.InferenceSession.create(await response.arrayBuffer(), {executionProviders:["wasm"]});
}

function clearInput() {
  generation++;
  clearTimeout(timer);
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  video.hidden = photo.hidden = true;
  photo.removeAttribute("src");
  if (photoURL) URL.revokeObjectURL(photoURL);
  photoURL = null;
  spoken = "";
  start.textContent = "Start camera";
  byId("object").textContent = byId("confidence").textContent = "—";
  return generation;
}

async function predict(session, source) {
  const w = source.videoWidth || source.naturalWidth;
  const h = source.videoHeight || source.naturalHeight;
  const side = Math.min(w, h);
  ctx.drawImage(source, (w-side)/2, (h-side)/2, side, side, 0, 0, 224, 224);
  const rgba = ctx.getImageData(0, 0, 224, 224).data;
  const input = new Float32Array(area*3);
  for (let i=0; i<area; i++) {
    input[i] = (rgba[4*i]/255 - .485)/.229;
    input[area+i] = (rgba[4*i+1]/255 - .456)/.224;
    input[2*area+i] = (rgba[4*i+2]/255 - .406)/.225;
  }
  const output = await session.run({[session.inputNames[0]]: new ort.Tensor("float32", input, [1,3,224,224])});
  const scores = output[session.outputNames[0]].data;
  const max = Math.max(...scores);
  let sum = 0;
  for (const score of scores) sum += Math.exp(score-max);
  return {label:imageLabels[scores.indexOf(max)], probability:1/sum};
}

async function recognize(source, id) {
  try {
    status.textContent = "Loading recognition model…";
    sessionPromise ??= loadModel().catch(error => {sessionPromise = null; throw error;});
    const session = await sessionPromise;
    async function next() {
      if (id !== generation) return;
      try {
        const {label, probability} = await predict(session, source);
        if (id !== generation) return;
        byId("object").textContent = label;
        byId("confidence").textContent = (probability * 100).toFixed(1) + "%";
        status.textContent = probability < .5 ? "Low confidence. Center one object in the frame." : "Prediction ready.";
        if (source === video) {
          timer = setTimeout(next, 300);
          if (probability > .5 && spoken !== label && window.speechSynthesis && window.SpeechSynthesisUtterance) {
            spoken = label;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(new SpeechSynthesisUtterance(label));
          }
        }
      } catch (error) {
        if (id === generation) status.textContent = "Prediction failed. " + error.message;
      }
    }
    await next();
  } catch (error) {
    if (id === generation) status.textContent = "Model startup failed. " + error.message;
  }
}

start.onclick = async () => {
  const wasRunning = !!stream;
  const id = clearInput();
  if (wasRunning) {status.textContent = "Camera stopped."; return;}
  status.textContent = "Waiting for camera access…";
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Open the HTTPS Pages URL or choose a photo.");
    const camera = await navigator.mediaDevices.getUserMedia({video:true, audio:false});
    if (id !== generation) {camera.getTracks().forEach(track => track.stop()); return;}
    stream = camera;
    camera.getVideoTracks()[0].onended = () => {
      if (id === generation) {clearInput(); status.textContent = "Camera disconnected. Reconnect it or choose a photo.";}
    };
    video.srcObject = stream;
    video.hidden = false;
    start.textContent = "Stop camera";
    await video.play();
    if (id === generation) recognize(video,id);
  } catch (error) {
    if (id !== generation) return;
    clearInput();
    status.textContent = error.name === "NotAllowedError" ? "Camera access was denied. Allow it in your browser or choose a photo." : "Camera unavailable. " + error.message + " You can choose a photo.";
  }
};

file.onchange = async () => {
  const selected = file.files[0];
  if (!selected) return;
  const id = clearInput();
  file.value = "";
  photoURL = URL.createObjectURL(selected);
  photo.src = photoURL;
  try {
    await photo.decode();
    if (id !== generation) return;
    photo.hidden = false;
    recognize(photo,id);
  } catch {
    if (id === generation) status.textContent = "The photo could not be opened. Try a JPEG or PNG.";
  }
};
start.disabled = file.disabled = false;
status.textContent = "Start your camera or choose a photo to identify an object.";
