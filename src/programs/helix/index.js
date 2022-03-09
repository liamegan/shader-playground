console.clear();

import {
  FragmentShader,
  Texture,
  Uniform,
} from "wtc-gl";

const shaderF = require("./helix.frag");

// Create the fragment shader wrapper
const FSWrapper = new FragmentShader({
  fragment: shaderF,
});

const { gl, uniforms } = FSWrapper;

// Create the texture
const texture = new Texture(gl, {
  wrapS: gl.REPEAT,
  wrapT: gl.REPEAT,
});
// Load the image into the uniform
const img = new Image();
img.crossOrigin = "anonymous";
img.src = "/img/noise.png";
img.onload = () => (texture.image = img);

uniforms.s_noise = new Uniform({
  name: "noise",
  value: texture,
  kind: "texture",
});
