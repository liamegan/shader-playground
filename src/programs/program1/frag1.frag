precision highp float;

uniform vec2 u_resolution;
uniform float u_time;
uniform vec4 u_mouse;
uniform sampler2D s_noise;

uniform sampler2D b_noise;

varying vec2 v_uv;
  
  /* Raymarching constants */
  /* --------------------- */
  const float MAX_TRACE_DISTANCE = 30.;             // max trace distance
  const float INTERSECTION_PRECISION = 0.001;       // precision of the intersection
  const int NUM_OF_TRACE_STEPS = 256;               // max number of trace steps
  const float STEP_MULTIPLIER = 1.;                 // the step mutliplier - ie, how much further to progress on each step
  
    
  const vec3 a = vec3(5.5, 1., -5.);
  const vec3 b = vec3(-1.5, -2., 5.);
  
  vec3 bg;
  vec2 mouse;
  
  /* Structures */
  /* ---------- */
  struct Camera {
    vec3 ro;
    vec3 rd;
    vec3 forward;
    vec3 right;
    vec3 up;
    float FOV;
  };
  struct Surface {
    float len;
    vec3 position;
    vec3 colour;
    float id;
    float steps;
    float AO;
  };
  struct Model {
    float dist;
    vec3 colour;
    float id;
  };
  
  vec2 getScreenSpace(vec2 from) {
    vec2 uv = (from - 0.5 * u_resolution.xy) / min(u_resolution.y, u_resolution.x);
    
    return uv;
  }
  
  //--------------------------------
  // Modelling
  //--------------------------------
  float dibox(vec3 p,vec3 b,vec3 rd){
    p/=b;
    vec3 dir = sign(rd)*.5;   
    vec3 rc = (dir-p)/rd;
    rc*=b;
    float dc = min(min(rc.x, rc.y), rc.z)+0.001;
    return dc;
  }
  float sdfRay(in vec3 p, in vec3 a, in vec3 b, in float r) {
    float h = clamp(dot(p-a, b-a)/dot(b-a, b-a), 0., 1.);
    return length(p-a-(b-a)*h)-r;
  }
  float sdBox( vec3 p, vec3 b ) {
    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
  }
  Model model(vec3 p, vec3 rd) {
    vec3 _p = p;
    p = mod(p, 2.) - 1.;
    
    float d = min(
      length(max(abs(p.xz) - .005, 0.)), 
      length(max(abs(p.xy) - .005, 0.)));
    d = min(d, length(max(abs(p.yz) - .005, 0.)));
    vec3 colour = vec3(1,0,0);
    
    float seg = sdfRay(_p, a, b, .02);
    
    if(seg < d) {
      d = seg;
      colour = vec3(0,1,0);
    }
    
    
    vec3 bpos = a - b;
    float bl = length(bpos);
    rd = normalize(bpos);
    bpos = b + bpos * clamp(mouse.y+.5, 0., 1.);
    float position = length(_p - bpos)-.08;
    
    if(position < d) {
      d = position;
      colour = vec3(1,1,0);
    }
    
    
//     float dist = 0.;
    vec3 lp = rd;
//     position = length(_p - (bpos + lp)) - .1;
    
//     if(position < d) {
//       d = position;
//       colour = vec3(0,1,1);
//     }
    
    
    vec3 q = mod(bpos-1., 2.) - 1.;           // dividing the ray position into domains
    vec3 dir = sign(rd);                      // taking the sign of the ray direction. If your cellspace is 1. then this should be halved etc.
    vec3 rC = (dir-q)/rd;	                    // ray to cell boundary
    float dC = min(min(rC.x, rC.y), rC.z);    // distance to cell boundary
    vec3 boxPos = _p - (bpos + lp * dC);      // The position at the boundary
    position = sdBox(boxPos, vec3(.1));
    
    if(position < d) {
      d = position;
      colour = vec3(0,1,1);
    }
    
    return Model(d, colour, 1.);
  }
  Model map( vec3 p, vec3 rd ) {
    return model(p, rd);
  }
  Model map( vec3 p) {
    vec3 rd = vec3(0);
    return map(p, rd);
  }
  
  
  Surface calcIntersection( in Camera cam ){
    float h =  INTERSECTION_PRECISION*2.0;
    float rayDepth = 0.0;
    float hitDepth = -1.0;
    float id = -1.;
    float steps = 0.;
    float ao = 0.;
    vec3 position;
    vec3 colour;

    for( int i=0; i< NUM_OF_TRACE_STEPS ; i++ ) {
      if( abs(h) < INTERSECTION_PRECISION || rayDepth > MAX_TRACE_DISTANCE ) break;
      position = cam.ro+cam.rd*rayDepth;
      Model m = map( position, cam.rd );
      h = m.dist;
      rayDepth += h * STEP_MULTIPLIER;
      id = m.id;
      steps += 1.;
      ao += max(h, 0.);
      colour = m.colour;
    }

    if( rayDepth < MAX_TRACE_DISTANCE ) hitDepth = rayDepth;
    if( rayDepth >= MAX_TRACE_DISTANCE ) id = -1.0;

    return Surface( hitDepth, position, colour, id, steps, ao );
  }
  Camera getCamera(in vec2 uv, in vec3 pos, in vec3 target) {
    vec3 forward = normalize(target - pos);
    vec3 right = normalize(vec3(forward.z, 0., -forward.x));
    vec3 up = normalize(cross(forward, right));
    
    float FOV = .6;
    
    return Camera(
      pos,
      normalize(forward + FOV * uv.x * right + FOV * uv.y * up),
      forward,
      right,
      up,
      FOV
    );
  }
  
  
  float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax ) {
    float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ ) {
      float h = map( ro + rd*t ).dist;
      res = min( res, 8.0*h/t );
      t += clamp( h, 0.02, 0.10 );
      if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
  }
  float calcAO( in vec3 pos, in vec3 nor ) {
    float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
      float hr = 0.01 + 0.12*float(i)/4.0;
      vec3 aopos =  nor * hr + pos;
      float dd = map( aopos ).dist;
      occ += -(dd-hr)*sca;
      sca *= 0.95;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 );    
  }
  vec3 shade(vec3 col, vec3 pos, vec3 nor, vec3 ref, Camera cam) {
    // lighitng        
    float occ = calcAO( pos, nor );
    vec3  lig = normalize( vec3(-0.6, 0.7, 0.) );
    float amb = clamp( 0.5+0.5*nor.y, 0.0, 1.0 );
    float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
    float bac = clamp( dot( nor, normalize(vec3(-lig.x,0.0,-lig.z))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
    //float dom = smoothstep( -0.1, 0.1, ref.y );
    float fre = pow( clamp(1.0+dot(nor,cam.rd),0.0,1.0), 2.0 );
    // float spe = pow(clamp( dot( ref, lig ), 0.0, 1.0 ),16.0);
    
    float fog = min(1. / (.02 * pow(length(cam.ro - pos), 2.)), 1.);

    // dif *= softshadow( pos, lig, 0.02, 2.5 );
    //dom *= softshadow( pos, ref, 0.02, 2.5 );

    vec3 lin = vec3(0.0);
    lin += 1.20*dif*vec3(.95,0.80,0.60);
    // lin += 1.20*spe*vec3(1.00,0.85,0.55)*dif;
    lin += 0.80*amb*vec3(0.50,0.70,.80)*occ;
    //lin += 0.30*dom*vec3(0.50,0.70,1.00)*occ;
    lin += 0.30*bac*vec3(0.25,0.25,0.25)*occ;
    lin += 0.20*fre*vec3(1.00,1.00,1.00)*occ;
    col = col*lin;
    
    col = mix(bg, col, fog);

    return col;
  }
  
  // Calculates the normal by taking a very small distance,
  // remapping the function, and getting normal for that
  vec3 calcNormal( in vec3 pos ){
    vec3 eps = vec3( 0.001, 0.0, 0.0 );
    vec3 nor = vec3(
      map(pos+eps.xyy).dist - map(pos-eps.xyy).dist,
      map(pos+eps.yxy).dist - map(pos-eps.yxy).dist,
      map(pos+eps.yyx).dist - map(pos-eps.yyx).dist );
    return normalize(nor);
  }
  
  vec3 render(Surface surface, Camera cam, vec2 uv) {
    vec3 colour = vec3(.04,.045,.05);
    colour = vec3(.35, .5, .65);
    vec3 colourB = vec3(.8, .8, .9);
    
    vec2 pp = uv;
    
    colour = mix(colourB, colour, length(pp)/1.5);// * clamp(mouse.y+.5, 0., 1.);
    bg = colour;

    if (surface.id == 1.){
      vec3 surfaceNormal = calcNormal( surface.position );
      vec3 ref = reflect(cam.rd, surfaceNormal);
      colour = surfaceNormal;
      colour = shade(surface.colour, surface.position, surfaceNormal, ref, cam);
    }

    return colour;
  }
  
  void main() {
    vec2 uv = getScreenSpace(gl_FragCoord.xy);
    mouse = getScreenSpace(u_mouse.xy);
    
    float t = u_time * 4.;
    t = 0.;
    float c = cos(t);
    float s = sin(t);
    float x = c*9.5;
    float z = s*9.5;
    
    // vec3 a = vec3(1.5, 1., -5.);
    // vec3 b = vec3(-1.5, -2., 5.);
    vec3 bpos = a - b;
    float bl = length(bpos);
    float y = clamp(mouse.y+.5, 0., 1.);
    bpos = b + bpos * y;
    
    // Camera cam = getCamera(uv, vec3(x,0,z), vec3(0.));
    Camera cam;
    if(u_mouse.z < 1.) {
      cam = getCamera(uv, vec3(x,0. + bpos.y + .2,z + bpos.z + .2), bpos);
    } else {
      cam = getCamera(uv, vec3(bpos.x - 1.35, bpos.y + .3, bpos.z + 2.), bpos);
    }
    
    Surface surface = calcIntersection(cam);
    
    gl_FragColor = vec4(render(surface, cam, uv), 1.);
  }