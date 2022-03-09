const nav = document.getElementById("nav");

document.body.addEventListener('keyup', (e) => {
  console.log(e.key);
  if('~`'.indexOf(e.key) > -1) {
    nav.classList.toggle('visible');
    e.preventDefault();
  }
})