/* ── Theme Toggle ── */
const html = document.documentElement;
const themeToggle = document.getElementById('theme-toggle');

function getStoredTheme() {
  return localStorage.getItem('rlm-theme');
}

function setStoredTheme(theme) {
  localStorage.setItem('rlm-theme', theme);
}

function applyTheme(theme) {
  html.setAttribute('data-theme', theme);
  setStoredTheme(theme);
}

function initTheme() {
  const stored = getStoredTheme();
  if (stored) {
    applyTheme(stored);
  } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    applyTheme('dark');
  } else {
    applyTheme('light');
  }
}

if (themeToggle) {
  themeToggle.addEventListener('click', () => {
    const current = html.getAttribute('data-theme');
    applyTheme(current === 'light' ? 'dark' : 'light');
  });
}

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
  if (!getStoredTheme()) {
    applyTheme(e.matches ? 'dark' : 'light');
  }
});

/* ── Sidebar Toggle (mobile) — optional, skip if no sidebar ── */
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebarOverlay = document.getElementById('sidebar-overlay');

if (sidebar && sidebarToggle && sidebarOverlay) {
  function openSidebar() {
    sidebar.classList.add('open');
    sidebarOverlay.classList.add('show');
    document.body.style.overflow = 'hidden';
  }

  function closeSidebar() {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('show');
    document.body.style.overflow = '';
  }

  sidebarToggle.addEventListener('click', () => {
    if (sidebar.classList.contains('open')) {
      closeSidebar();
    } else {
      openSidebar();
    }
  });

  sidebarOverlay.addEventListener('click', closeSidebar);

  /* Close sidebar when clicking a link (mobile) */
  sidebar.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      if (window.innerWidth <= 900) {
        closeSidebar();
      }
    });
  });
}

/* ── Reading Progress Bar ── */
const progressBar = document.getElementById('progress-bar');

function updateProgress() {
  const scrollTop = window.scrollY;
  const docHeight = document.documentElement.scrollHeight - window.innerHeight;
  const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
  progressBar.style.width = progress + '%';
}

window.addEventListener('scroll', updateProgress, { passive: true });

/* ── Active Section Highlighting ── */
const sections = document.querySelectorAll('section[id]');
const navLinks = sidebar ? sidebar.querySelectorAll('.sidebar-section a[href^="#"]') : [];

if (navLinks.length > 0) {
  function updateActiveNav() {
    const scrollPos = window.scrollY + 100;

    let current = '';
    sections.forEach(section => {
      if (section.offsetTop <= scrollPos) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === '#' + current) {
        link.classList.add('active');
      }
    });
  }

  window.addEventListener('scroll', updateActiveNav, { passive: true });
}

/* ── Smooth reveal on scroll ── */
const observerOptions = {
  root: null,
  rootMargin: '0px 0px -60px 0px',
  threshold: 0.1
};

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
    }
  });
}, observerOptions);

document.querySelectorAll('section').forEach(section => {
  section.style.opacity = '0';
  section.style.transform = 'translateY(20px)';
  section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
  observer.observe(section);
});

/* Make title section visible immediately */
const titleSection = document.querySelector('.title-section');
if (titleSection) {
  titleSection.style.opacity = '1';
  titleSection.style.transform = 'translateY(0)';
}

/* Make notes-hero visible immediately */
const notesHero = document.querySelector('.notes-hero');
if (notesHero) {
  notesHero.style.opacity = '1';
  notesHero.style.transform = 'translateY(0)';
}

/* ── Init ── */
initTheme();
if (navLinks.length > 0) {
  updateActiveNav();
}
