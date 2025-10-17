// Mobile menu toggle
document.addEventListener('DOMContentLoaded', function() {
  var menuIcon = document.querySelector('.menu-icon');
  var trigger = document.querySelector('.site-nav .trigger');

  if (menuIcon && trigger) {
    menuIcon.addEventListener('click', function(e) {
      e.preventDefault();
      if (trigger.style.display === 'block') {
        trigger.style.display = 'none';
      } else {
        trigger.style.display = 'block';
      }
    });
  }
});
