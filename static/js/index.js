$(document).ready(function () {
  // Carousels (used by _includes/carousel.html)
  if (window.bulmaCarousel) {
    document.querySelectorAll('.carousel').forEach(function (carousel) {
      bulmaCarousel.attach(carousel, {
        slidesToScroll: 1,
        slidesToShow: parseInt(carousel.dataset.slides || '3', 10),
        loop: true,
        autoplay: carousel.dataset.autoplay === 'true',
        autoplaySpeed: parseInt(carousel.dataset.autoplaySpeed || '3000', 10),
        pauseOnHover: true
      });
    });
  }

  // "Copy" button on the BibTeX block
  $('.copy-bibtex').on('click', function () {
    var button = $(this);
    var text = $('#bibtex-code').text();
    navigator.clipboard.writeText(text).then(function () {
      button.find('span:last').text('Copied');
      setTimeout(function () { button.find('span:last').text('Copy'); }, 1500);
    });
  });
});
