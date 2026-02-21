// Task list for SO-101 Task Set
const tasks = [
  'Reach Cube', 'Reach Can', 'Lift Cube', 'Lift Can',
  'Place Cube', 'Place Can', 'Stack Cube', 'Stack Can'
];

// Video paths per task
const taskVideos = {
  'Reach Cube':  { sim: './static/videos/sim/reach_cube.mp4', real: './static/videos/real/reach_cube.mp4' },
  'Reach Can':   { sim: './static/videos/sim/reach_can.mp4',  real: './static/videos/real/reach_can.mp4' },
  'Lift Cube':   { sim: './static/videos/sim/lift_cube.mp4',  real: './static/videos/real/lift_cube.mp4' },
  'Lift Can':    { sim: './static/videos/sim/lift_can.mp4',   real: './static/videos/real/lift_can.mp4' },
  'Place Cube':  { sim: './static/videos/sim/place_cube.mp4', real: './static/videos/real/place_cube.mp4' },
  'Place Can':   { sim: './static/videos/sim/place_can.mp4',  real: './static/videos/real/place_can.mp4' },
  'Stack Cube':  { sim: './static/videos/sim/stack_cube.mp4', real: './static/videos/real/stack_cube.mp4' },
  'Stack Can':   { sim: './static/videos/sim/stack_can.mp4',  real: './static/videos/real/stack_can.mp4' },
};

let currentTaskIndex = 7;

// Function to switch the task tab
function switchTask(taskIndex) {
  if (currentTaskIndex === taskIndex) return;

  // Lock scroll position to prevent jump during video load
  const scrollY = window.scrollY;

  // Update active tab styling
  const taskTabs = document.getElementById('task-tabs').querySelectorAll('.tab');
  taskTabs.forEach((tab, index) => {
    tab.classList.toggle('active', index === taskIndex);
  });

  currentTaskIndex = taskIndex;
  const task = tasks[taskIndex];

  // Update video sources
  const simVideo = document.getElementById('sim-video');
  const realVideo = document.getElementById('real-video');

  simVideo.src = taskVideos[task].sim;
  realVideo.src = taskVideos[task].real;

  simVideo.load();
  simVideo.play().catch(e => console.warn('Auto-play prevented:', e));
  realVideo.load();
  realVideo.play().catch(e => console.warn('Auto-play prevented:', e));

  // Restore scroll position after load
  window.scrollTo(0, scrollY);
  requestAnimationFrame(() => window.scrollTo(0, scrollY));
}

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', function() {
  currentTaskIndex = 7;

  // Back to Top Button
  const backToTopButton = document.getElementById('back-to-top');
  if (backToTopButton) {
    backToTopButton.addEventListener('click', function() {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }
});
