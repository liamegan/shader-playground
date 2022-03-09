console.clear();

import {
  Vec2,
  Vec3,
  Vec4,
  Mat2,
  Mat3,
  Mat4,
  Quat,
} from "wtc-math";
import {
  FragmentShader,
  Texture,
  Uniform,
} from "wtc-gl";

// const shaderF = document.querySelector("#fragShader").innerText;
const shaderF = require("./frag1.frag");
console.log(shaderF);

// Create the fragment shader wrapper
const FSWrapper = new FragmentShader({
  fragment: shaderF,
});

const { gl, uniforms, renderer } = FSWrapper;
const px = renderer.dpr;

// Create the texture
const texture = new Texture(gl, {
  wrapS: gl.REPEAT,
  wrapT: gl.REPEAT,
});
// Load the image into the uniform
const img = new Image();
img.crossOrigin = "anonymous";
img.src = "https://assets.codepen.io/982762/noise.png";
img.onload = () => (texture.image = img);

uniforms.s_noise = new Uniform({
  name: "noise",
  value: texture,
  kind: "texture",
});

// Set up mouse uniforms
(function () {
  const tarmouse = new Vec4(0, 0, 0, 0);
  const curmouse = tarmouse.clone();
  let pointerdown = false;
  uniforms.u_mouse = new Uniform({
    name: "mouse",
    value: tarmouse.array,
    kind: "float_vec4",
  });
  uniforms.u_mouselength = new Uniform({
    name: "mouselength",
    value: 0,
    kind: "float",
  });
  document.body.addEventListener("pointermove", (e) => {
    tarmouse.x = e.x * px;
    tarmouse.y = (window.innerHeight - e.y) * px;
    // if(pointerdown) {
    //   tarmouse.z = e.x*px;
    //   tarmouse.w = ( window.innerHeight - e.y ) * px;
    // }
  });
  document.body.addEventListener("pointerdown", (e) => {
    tarmouse.z = 1;
    curmouse.z = 1;
    // pointerdown = true;
    // tarmouse.z = e.x*px;
    // tarmouse.w = ( window.innerHeight - e.y ) * px;
  });
  document.body.addEventListener("pointerup", (e) => {
    tarmouse.z = 0;
    curmouse.z = 0;
  });
  let oldTime;
  const animouse = (d) => {
    const factor = d - oldTime;
    oldTime = d;
    const diff = tarmouse.subtractNew(curmouse);
    uniforms.u_mouselength.value = diff.length;
    curmouse.add(diff.scale((1 / factor) * 0.5));
    uniforms.u_mouse.value = curmouse.array;
    requestAnimationFrame(animouse);
  };
  requestAnimationFrame(animouse);
})();
