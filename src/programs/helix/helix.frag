precision highp float;

uniform vec2 u_resolution;
uniform float u_time;
uniform vec2 u_mouse;
uniform sampler2D s_noise;

uniform sampler2D b_noise;

varying vec2 v_uv;
  
#define PI 3.14159265359
  
  /* Raymarching constants */
  /* --------------------- */
  const float MAX_TRACE_DISTANCE = 10.;             // max trace distance
  const float INTERSECTION_PRECISION = 0.001;       // precision of the intersection
  const int NUM_OF_TRACE_STEPS = 256;               // max number of trace steps
  const float STEP_MULTIPLIER = .5;                 // the step mutliplier - ie, how much further to progress on each step
  
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
  
  vec2 getScreenSpace() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / min(u_resolution.y, u_resolution.x);
    
    return uv;
  }
  
  void pR(inout vec2 p, float a) {
      p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
  }
  
  // Repeat space along one axis
  float pMod1(inout float p, float size) {
      float halfsize = size*0.5;
      float c = floor((p + halfsize)/size);
      p = mod(p + halfsize, size) - halfsize;
      return c;
  }

  
  // Cartesian to polar coordinates
  vec3 cartToPolar(vec3 p) {
      float x = p.x; // distance from the plane it lies on
      float a = atan(p.y, p.z); // angle around center
      float r = length(p.zy); // distance from center
      return vec3(x, a, r);
  }

  // Polar to cartesian coordinates
  vec3 polarToCart(vec3 p) {
      return vec3(
          p.x,
          sin(p.y) * p.z,
          cos(p.y) * p.z
      );
  }

  // Closest of two points
  vec3 closestPoint(vec3 pos, vec3 p1, vec3 p2) {
      if (length(pos - p1) < length(pos - p2)) {
          return p1;
      } else {
          return p2;
      }
  }

  // http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
  mat3 rotationMatrix(vec3 axis, float angle)
  {
      axis = normalize(axis);
      float s = sin(angle);
      float c = cos(angle);
      float oc = 1.0 - c;

      return mat3(
          oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
          oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
          oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c
      );
  }
  
  // --------------------------------------------------------
  // Helix
  // --------------------------------------------------------

  vec2 closestPointOnRepeatedLine(vec2 line, vec2 point){

      // Angle of the line
      float a = atan(line.x, line.y);

      // Rotate space so we can easily repeat along
      // one dimension
      pR(point, -a);

      // Repeat to create parallel lines at the corners
      // of the vec2(lead, radius) polar bounding area
      float repeatSize = sin(a) * line.y;
      float cell = pMod1(point.x, repeatSize);

      // Rotate space back to where it was
      pR(point, a);

      // Closest point on a line
      line = normalize(line);
      float d = dot(point, line);
      vec2 closest = line * d;

      // Part 2 of the repeat, move the line along it's
      // perpendicular by the repeat cell
      vec2 perpendicular = vec2(line.y, -line.x);
      closest += cell * repeatSize * perpendicular;

      return closest;
  }

  // Closest point on a helix
  vec3 closestHelix(vec3 p, float lead, float radius) {

      p = cartToPolar(p);
      p.y *= radius;

      vec2 line = vec2(lead, radius * PI * 2.);
      vec2 closest = closestPointOnRepeatedLine(line, p.xy);

      closest.y /= radius;
      vec3 closestCart = polarToCart(vec3(closest, radius));

      return closestCart;
  }


  // Cartesian to helix coordinates
  vec3 helixCoordinates(vec3 p, vec3 closest, float lead, float radius) {
      float helixAngle = atan((2. * PI * radius) / lead);
      vec3 normal = normalize(closest - vec3(closest.x,0,0));
      vec3 tangent = vec3(1,0,0) * rotationMatrix(normal, helixAngle);
      float x = (closest.x / lead) * radius * PI * 2.;
      float y = dot(p - closest, cross(tangent, normal));
      float z = dot(p - closest, normal);
      return vec3(x,y,z);
  }
  
  //--------------------------------
  // Modelling
  //--------------------------------
  Model model(vec3 p) {
    float d = length(p) - .2;
    p.x -= u_time;
    
    vec3 helix = closestHelix(p, .4, .1);
    d = length(p - helix) - .01;
    p = helixCoordinates(p, helix, .2, .05);
    
    helix = closestHelix(p, .05, .03);
    d = min(d, length(p - helix) - .002);
    
    vec3 colour = vec3(.5);
    return Model(d, colour, 1.);
  }
  Model map( vec3 p ){
    return model(p);
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
      Model m = map( position );
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
  vec3 shade(vec3 col, vec3 pos, vec3 nor, vec3 ref, Camera cam, Surface surface) {
    // lighitng        
    float occ = calcAO( pos, nor );
    vec3  lig = normalize( vec3(-0.0, 0.2, -0.2) - pos );
    float amb = clamp( 0.5+0.5*nor.y, 0.0, 1.0 );
    float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
    float bac = clamp( dot( nor, normalize(vec3(-lig.x,0.0,-lig.z))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
    //float dom = smoothstep( -0.1, 0.1, ref.y );
    float fre = pow( clamp(1.0+dot(nor,cam.rd),0.0,1.0), 2.0 );
    // float spe = pow(clamp( dot( ref, lig ), 0.0, 1.0 ),16.0);

    // dif *= softshadow( pos, lig, 0.02, 2.5 );
    dif *= 1./surface.AO;
    // dom *= softshadow( pos, ref, 0.02, 2.5 );

    vec3 lin = vec3(0.0);
    lin += 1.20*dif*vec3(.95,0.80,0.60);
    // lin += 1.20*spe*vec3(1.00,0.85,0.55)*dif;
    lin += 0.80*amb*vec3(0.50,0.70,.80)*occ;
    //lin += 0.30*dom*vec3(0.50,0.70,1.00)*occ;
    lin += 0.30*bac*vec3(0.25,0.25,0.25)*occ;
    lin += 0.20*fre*vec3(1.00,1.00,1.00)*occ;
    col = col*lin;
    
    // return vec3(surface.AO);

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
    vec3 colour = vec3(.04,.04,.04);
    colour = vec3(.45, .45, .5);
    vec3 colourB = vec3(.9, .9, .9);
    
    vec2 pp = uv;
    
    colour = mix(colourB, colour, length(pp)/1.5);

    if (surface.id == 1.){
      vec3 surfaceNormal = calcNormal( surface.position );
      vec3 ref = reflect(cam.rd, surfaceNormal);
      colour = surfaceNormal;
      colour = shade(surface.colour, surface.position, surfaceNormal, ref, cam, surface);
    }

    return colour;
  }
  
  void main() {
    vec2 uv = getScreenSpace();
    
    Camera cam = getCamera(uv, vec3(-.0,0,.8), vec3(.2, 0, 0));
    
    Surface surface = calcIntersection(cam);
    
    gl_FragColor = vec4(render(surface, cam, uv), 1.);
  }