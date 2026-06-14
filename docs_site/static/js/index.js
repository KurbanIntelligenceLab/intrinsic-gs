


function main() {
  setupTabs();
  setupModals();
  setupImageLoading();
}

function setupImageLoading() {
  const groups = [
    'iphone',
    'dalle2',
    'co3d',
    'imagenet',
    'hypersim',
    'taskonomy',
  ];

  // When we click on a tab, schedule its images for loading.
  for (let group of groups) {
    $('li[data-tab="' + group + '"]').click(() => {
      loadImages(group);
    });
  }

  // Start loading image groups in order.
  // This is pretty ugly! I should learn how to use promises to avoid these
  // nasty nested callbacks!
  loadImages('iphone', function() {
    loadImages('dalle2', function() {
      loadImages('co3d', function() {
        loadImages('imagenet', function() {
          loadImages('hypersim', function() {
            loadImages('taskonomy');
          });
        });
      });
    });
  });
}

let _loaded_groups = new Set();

// Calling this schedules images in the specified group for loading by copying
// their data-src attribute to src; once all images in the group have loaded
// we call the callback (which defaults to a trivial function).
function loadImages(group, cb = () => {}) {
  // Make sure we don't try to load the same group more than once
  if (_loaded_groups.has(group)) {
    console.log('images in group ' + group + ' have already been scheduled to load');
    return;
  }
  console.log('scheduling images in group ' + group);
  _loaded_groups.add(group);

  // Find all images in the group
  let imgs = $('div[data-content="' + group + '"] img');

  let to_load = new Set();
  imgs.each((idx, img) => {
    const url = $(img).data('src');
    to_load.add(url);
    img.onload = function() {
      to_load.delete(url);
      if (to_load.size === 0) {
        cb();
      }
    }
  });

  imgs.each((idx, img) => {
    const url = $(img).data('src');
    img.src = url;
  });
}


function openModal(elem) {
  $(elem).addClass('is-active');
  $('#plot-loading-div').show();
  $('#plot-div').hide();
}


function closeModal(elem) {
  $(elem).removeClass('is-active');
  $('#plot-div').empty();
}


function setupModals() {
  // Add click events on buttons to open the modal
  $('.js-modal-trigger').click(function() {
    const modal_id = $(this).data('target');
    const modal = document.getElementById(modal_id);
    const scene_dir = 'static/resources/' + $(this).data('path');
    const pc_path = scene_dir + '/pointcloud.json';
    const fmt_path = scene_dir + '/fmt.json';
    openModal(modal);
    $.getJSON(pc_path, function(data) {
      // console.log(data);
      const plot_data = [
        {
          type: "scatter3d",
          mode: "markers",
          x: data.x,
          y: data.y,
          z: data.z,
          marker: {
            size: 3,
            color: data.color,
          }
        }
      ];
      const layout = {
        height: 800,
        width: 800,
        margin: {
          l: 0,
          r: 0,
        },
        scene: {
          aspectmode: 'cube',
          xaxis: {
            showgrid: false,
            zeroline: false,
            showline: false,
            showticklabels: false,
            showaxeslabels: false,
          },
          yaxis: {
            showgrid: false,
            zeroline: false,
            showline: false,
            showticklabels: false,
            showaxeslabels: false,
          },
          zaxis: {
            showgrid: false,
            zeroline: false,
            showline: false,
            showticklabels: false,
            showaxeslabels: false,
          },
          camera: {
            up: {
              x: 0.0,
              y: 1.0,
              z: 0.0,
            }
          }
        }
      }
      $('#plot-loading-div').hide();
      $('#plot-div').show();
      Plotly.newPlot('plot-div', plot_data, layout);
    });
  });

  $('.modal-background,.modal-close,.modal-card-head,.delete,.modal-card-front').click(function() {
    const modal = this.closest('.modal');
    closeModal(modal);
  });
}


function setupTabs() {
  // Set up click handlers for the tabs
  $('#tabs li').on('click', function() {
    let tab = $(this).data('tab');

    $('#tabs li').removeClass('is-active');
    $(this).addClass('is-active');

    $('#tabs-content div').removeClass('is-active');
    $('div[data-content="' + tab + '"]').addClass('is-active');
  });
}

$(document).ready(main);
