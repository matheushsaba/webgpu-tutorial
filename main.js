// Assuming you have a way to create matrices (e.g., using gl-matrix or similar)
function createPerspectiveMatrix() {
    // Create a perspective projection matrix
    // This is just an example; parameters need to be adjusted for your specific needs
    let fov = 60 * (Math.PI / 180); // Field of view 60 degrees
    let aspect = canvas.clientWidth / canvas.clientHeight;
    let near = 0.1;
    let far = 100.0;
    return mat4.perspective(mat4.create(), fov, aspect, near, far);
  }
  
  function createModelViewMatrix(rotationY, rotationZ) {
    // Create a model-view matrix to position and rotate the cube
    let modelViewMatrix = mat4.create();
    mat4.translate(modelViewMatrix, modelViewMatrix, [0, 0, -3]); // Move back in Z so we can see the cube
    mat4.rotateY(modelViewMatrix, modelViewMatrix, rotationY); // Rotate around the Y axis
    mat4.rotateZ(modelViewMatrix, modelViewMatrix, rotationZ); // Rotate around the Y axis
    return modelViewMatrix;
  }

const canvas = document.querySelector("canvas");

// WebGPU device initialization
if (!navigator.gpu) {
  throw new Error("WebGPU not supported on this browser.");
}

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  throw new Error("No appropriate GPUAdapter found.");
}

const device = await adapter.requestDevice();

// Canvas configuration
const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat,
});

// Define vertices for a cube
const vertices = new Float32Array([
    // X, Y, Z coordinates
    -0.5, -0.5,  0.5, // Vertex 0
     0.5, -0.5,  0.5, // Vertex 1
     0.5,  0.5,  0.5, // Vertex 2
    -0.5,  0.5,  0.5, // Vertex 3
    -0.5, -0.5, -0.5, // Vertex 4
     0.5, -0.5, -0.5, // Vertex 5
     0.5,  0.5, -0.5, // Vertex 6
    -0.5,  0.5, -0.5, // Vertex 7
  ]);
  const vertexBuffer = device.createBuffer({
    label: "Cube vertices",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertices);

// Define indices for the cube
const indices = new Uint32Array([
    0, 1, 2, 2, 3, 0, // Front face
    1, 5, 6, 6, 2, 1, // Right face
    5, 4, 7, 7, 6, 5, // Back face
    4, 0, 3, 3, 7, 4, // Left face
    3, 2, 6, 6, 7, 3, // Top face
    4, 5, 1, 1, 0, 4  // Bottom face
  ]);
  const indexBuffer = device.createBuffer({
    label: "Cube indices",
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indexBuffer, 0, indices);

  // Update the vertex buffer layout for 3D coordinates
  const vertexBufferLayout = {
    arrayStride: 4 * 3, // 3 floats per vertex, 4 bytes per float
    attributes: [{
      format: "float32x3",
      offset: 0,
      shaderLocation: 0, // Position. Matches @location(0) in the @vertex shader.
    }],
  };

  const mvpMatrixBuffer = device.createBuffer({
    size: 64, // 4x4 matrix of float32 values
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  
// Update your vertex shader to accept a uniform MVP matrix
const cellShaderModule = device.createShaderModule({
    label: "Cube shader",
    code: `
      struct Uniforms {
        mvpMatrix : mat4x4<f32>,
      };
      @group(0) @binding(0) var<uniform> uniforms : Uniforms;
  
      @vertex
      fn vertexMain(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
        return uniforms.mvpMatrix * vec4<f32>(position, 1.0);
      }
  
      @fragment
      fn fragmentMain() -> @location(0) vec4<f32> {
        return vec4<f32>(1, 0, 0, 1); // Red color
      }
    `
  });

  const lineIndices = new Uint32Array([
    // Each pair of vertices forms an edge
    0, 1, 1, 2, 2, 3, 3, 0, // Front face
    4, 5, 5, 6, 6, 7, 7, 4, // Back face
    0, 4, 1, 5, 2, 6, 3, 7, // Connecting edges
]);

const lineIndexBuffer = device.createBuffer({
    label: "Cube line indices",
    size: lineIndices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(lineIndexBuffer, 0, lineIndices);


  const lineShaderModule = device.createShaderModule({
    label: "Line shader",
    code: `
      struct Uniforms {
        mvpMatrix : mat4x4<f32>,
      };
      @group(0) @binding(0) var<uniform> uniforms : Uniforms;
  
      @vertex
      fn vertexMain(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
        return uniforms.mvpMatrix * vec4<f32>(position, 1.0);
      }
  
      @fragment
      fn fragmentMain() -> @location(0) vec4<f32> {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0); // Output white color
      }
    `
  });
  
  // You'll also need to create a uniform buffer for the MVP matrix and update it on each frame
  let mvpMatrix = createPerspectiveMatrix(); // Initially, just the perspective matrix
  let modelViewMatrix = createModelViewMatrix(0); // Placeholder, will update per frame
  mat4.multiply(mvpMatrix, mvpMatrix, modelViewMatrix); // Combine the model-view and projection matrices
  
  const mvpBuffer = device.createBuffer({
    size: 64, // 4x4 matrix of floats
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(mvpBuffer, 0, new Float32Array(mvpMatrix));
  
  // Update your pipeline to include the uniform bind group for the MVP matrix
  const uniformBindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: {
        type: 'uniform',
      },
    }],
  });
  
  const uniformBindGroup = device.createBindGroup({
    layout: uniformBindGroupLayout,
    entries: [{
      binding: 0,
      resource: {
        buffer: mvpBuffer,
      },
    }],
  });
  
  const cellPipeline = device.createRenderPipeline({
    label: "Cube pipeline",
    layout: device.createPipelineLayout({ bindGroupLayouts: [uniformBindGroupLayout] }),
    vertex: {
      module: cellShaderModule,
      entryPoint: "vertexMain",
      buffers: [vertexBufferLayout], // Ensure this is correctly defined earlier
    },
    fragment: {
      module: cellShaderModule,
      entryPoint: "fragmentMain",
      targets: [{
        format: canvasFormat,
      }],
    },
    primitive: {
      topology: 'triangle-list', // Define the primitive topology
      cullMode: 'back', // Optional: cull back faces
    },
    depthStencil: { // Define depth-stencil attachment for 3D rendering
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus-stencil8',
    },
  });

  const linePipeline = device.createRenderPipeline({
    label: "Line pipeline",
    layout: device.createPipelineLayout({ bindGroupLayouts: [uniformBindGroupLayout] }),
    vertex: {
      module: lineShaderModule,
      entryPoint: "vertexMain",
      buffers: [vertexBufferLayout],
    },
    fragment: {
      module: lineShaderModule,
      entryPoint: "fragmentMain",
      targets: [{
        format: canvasFormat,
      }],
    },
    primitive: {
      topology: 'line-list',
      cullMode: 'none',
    },
    // Add this depth-stencil configuration to match the render pass expectation
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus-stencil8',
    },
  });
  
  
// Create a depth-stencil texture
const depthStencilTexture = device.createTexture({
    size: {
      width: canvas.width,
      height: canvas.height,
      depthOrArrayLayers: 1,
    },
    format: 'depth24plus-stencil8',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Animation loop to rotate the cube
  function animate(now) {
    now *= 0.001; // Convert time to seconds
    const rotationY = now/2; // Simple linear rotation over time
    const rotationZ = now/8; // Simple linear rotation over time
  
    modelViewMatrix = createModelViewMatrix(rotationY, rotationZ);
    mat4.multiply(mvpMatrix, createPerspectiveMatrix(), modelViewMatrix); // Update MVP matrix
    device.queue.writeBuffer(mvpBuffer, 0, new Float32Array(mvpMatrix));
  
    // Redraw the cube
    const encoder = device.createCommandEncoder();
    const passDescriptor = {
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear',
          clearValue: { r: 0, g: 0, b: 0, a: 1 }, // Clear to black, fully opaque
          storeOp: 'store',
        }],
        depthStencilAttachment: { // Add this configuration for depth-stencil attachment
          view: depthStencilTexture.createView(),
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
          stencilClearValue: 0,
          stencilLoadOp: 'clear',
          stencilStoreOp: 'store',
        }
      };
    const pass = encoder.beginRenderPass(passDescriptor);
    pass.setPipeline(cellPipeline);
    pass.setBindGroup(0, uniformBindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setIndexBuffer(indexBuffer, 'uint32');
    pass.drawIndexed(indices.length);
    
    pass.setPipeline(linePipeline); // Switch to the pipeline configured for drawing lines
    pass.setIndexBuffer(lineIndexBuffer, 'uint32'); // Use the line index buffer
    pass.drawIndexed(lineIndices.length); // Draw the lines based on the lineIndices
    
    pass.end();
    device.queue.submit([encoder.finish()]);
  
    requestAnimationFrame(animate); // Request the next frame
  }
  
  requestAnimationFrame(animate); // Start the animation loop