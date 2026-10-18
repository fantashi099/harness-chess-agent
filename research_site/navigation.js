// Keep the article section when crossing into the existing event viewer.
const blogLink = document.getElementById('blog-return');
try {
  const section = sessionStorage.getItem('chess-blog-section');
  if (blogLink && /^[a-z-]+$/.test(section ?? '')) blogLink.href = `index.html#${section}`;
} catch { /* Navigation remains usable when storage is blocked. */ }
if (document.body.classList.contains('blog-page')) {
  document.addEventListener('click', event => {
    const link = event.target.closest('a');
    if (!link || !link.getAttribute('href')?.startsWith('viewer.html')) return;
    const section = link.closest('[data-blog-section]')?.dataset.blogSection ?? link.closest('.article-body section')?.id ?? 'article';
    try {sessionStorage.setItem('chess-blog-section', section);} catch { /* Anchor fallback remains available. */ }
  });
}
