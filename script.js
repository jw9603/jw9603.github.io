// ===== Constants =====
const TYPING_DELETE_DELAY = 50;
const TYPING_ADD_DELAY = 100;
const TYPING_PAUSE_DELAY = 2000;
const TYPING_NEXT_DELAY = 500;
const PARTICLE_COUNT = 80;
const COUNTER_STEPS = 40;
const COUNTER_INTERVAL = 50;

const typingTexts = [
    'AI Solutions',
    'MCP Servers',
    'LLM Applications',
    'FastAPI Services'
];

// ===== Typing Animation =====
let textIndex = 0;
let charIndex = 0;
let isDeleting = false;

function type() {
    const typingElement = document.getElementById('typingText');
    if (!typingElement) return;

    const currentText = typingTexts[textIndex];

    if (isDeleting) {
        typingElement.textContent = currentText.substring(0, charIndex - 1);
        charIndex--;
    } else {
        typingElement.textContent = currentText.substring(0, charIndex + 1);
        charIndex++;
    }

    let delay = isDeleting ? TYPING_DELETE_DELAY : TYPING_ADD_DELAY;

    if (!isDeleting && charIndex === currentText.length) {
        delay = TYPING_PAUSE_DELAY;
        isDeleting = true;
    } else if (isDeleting && charIndex === 0) {
        isDeleting = false;
        textIndex = (textIndex + 1) % typingTexts.length;
        delay = TYPING_NEXT_DELAY;
    }

    setTimeout(type, delay);
}

// ===== Particles =====
function createParticles() {
    const container = document.getElementById('particles');
    if (!container) return;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const particle = document.createElement('div');
        particle.classList.add('particle');
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 10 + 's';
        particle.style.animationDuration = (6 + Math.random() * 8) + 's';
        container.appendChild(particle);
    }
}

// ===== Navigation Scroll Effect =====
function handleNavScroll() {
    const nav = document.getElementById('nav');
    if (!nav) return;
    nav.classList.toggle('scrolled', window.scrollY > 50);
}

// ===== Mobile Nav Toggle =====
function setupMobileNav() {
    const toggle = document.getElementById('navToggle');
    const links = document.querySelector('.nav-links');
    if (!toggle || !links) return;

    toggle.addEventListener('click', () => {
        links.classList.toggle('active');
        toggle.classList.toggle('active');
    });

    links.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            links.classList.remove('active');
            toggle.classList.remove('active');
        });
    });
}

// ===== Unified Scroll Observer =====
function setupScrollObservers() {
    // Scroll reveal
    const revealElements = document.querySelectorAll(
        '.skill-card, .project-card, .timeline-item, .about-grid, .contact-content, .pub-card'
    );
    revealElements.forEach(el => el.classList.add('reveal'));

    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                revealObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    revealElements.forEach(el => revealObserver.observe(el));

    // Skill bars + counters (shared observer)
    const animateElements = [
        ...document.querySelectorAll('.skill-fill'),
        ...document.querySelectorAll('.stat-number')
    ];

    const animateObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;

            if (entry.target.classList.contains('skill-fill')) {
                const width = entry.target.getAttribute('data-width');
                entry.target.style.width = width + '%';
            } else if (entry.target.classList.contains('stat-number')) {
                const target = parseInt(entry.target.getAttribute('data-target'));
                if (target) animateCounter(entry.target, target);
            }

            animateObserver.unobserve(entry.target);
        });
    }, { threshold: 0.5 });

    animateElements.forEach(el => animateObserver.observe(el));
}

// ===== Counter Animation =====
function animateCounter(element, target) {
    let current = 0;
    const increment = Math.max(target / COUNTER_STEPS, 1);
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = target;
            clearInterval(timer);
        } else {
            element.textContent = Math.ceil(current);
        }
    }, COUNTER_INTERVAL);
}

// ===== Smooth Scroll for Nav Links =====
function setupSmoothScroll() {
    document.querySelectorAll('.nav-links a[href^="#"], .hero-cta a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

// ===== Theme Toggle =====
function setupThemeToggle() {
    const toggle = document.getElementById('themeToggle');
    if (!toggle) return;

    const saved = localStorage.getItem('theme');
    if (saved) {
        document.documentElement.setAttribute('data-theme', saved);
    } else if (window.matchMedia('(prefers-color-scheme: light)').matches) {
        document.documentElement.setAttribute('data-theme', 'light');
    }

    toggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
    });
}

// ===== Initialize =====
document.addEventListener('DOMContentLoaded', () => {
    setupThemeToggle();
    createParticles();
    type();
    setupMobileNav();
    setupScrollObservers();
    setupSmoothScroll();
    window.addEventListener('scroll', handleNavScroll);
});
