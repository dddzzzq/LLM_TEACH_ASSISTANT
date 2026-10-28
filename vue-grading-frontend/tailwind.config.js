/** @type {import('tailwindcss').Config} */
const scale = (prefix) =>
  Object.fromEntries(
    [50, 100, 200, 300, 400, 500, 600, 700, 800, 900].map((n) => [
      n,
      `rgb(var(--color-${prefix}-${n}-rgb) / <alpha-value>)`,
    ])
  );
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        white: 'rgb(var(--color-white-rgb) / <alpha-value>)',
        black: 'rgb(var(--color-ink-rgb) / <alpha-value>)',
        gray: scale('gray'),
        slate: scale('gray'),
        indigo: scale('accent'),
        blue: scale('accent'),
        green: {
          50: 'rgb(var(--color-success-soft-rgb) / <alpha-value>)',
          100: 'rgb(var(--color-success-soft-rgb) / <alpha-value>)',
          500: 'rgb(var(--color-success-rgb) / <alpha-value>)',
          600: 'rgb(var(--color-success-rgb) / <alpha-value>)',
          700: 'rgb(var(--color-success-rgb) / <alpha-value>)',
          800: 'rgb(var(--color-success-rgb) / <alpha-value>)',
        },
        red: {
          50: 'rgb(var(--color-danger-soft-rgb) / <alpha-value>)',
          100: 'rgb(var(--color-danger-soft-rgb) / <alpha-value>)',
          500: 'rgb(var(--color-danger-rgb) / <alpha-value>)',
          600: 'rgb(var(--color-danger-rgb) / <alpha-value>)',
          700: 'rgb(var(--color-danger-rgb) / <alpha-value>)',
          800: 'rgb(var(--color-danger-rgb) / <alpha-value>)',
        },
        yellow: {
          50: 'rgb(var(--color-warning-soft-rgb) / <alpha-value>)',
          100: 'rgb(var(--color-warning-soft-rgb) / <alpha-value>)',
          500: 'rgb(var(--color-warning-rgb) / <alpha-value>)',
          600: 'rgb(var(--color-warning-rgb) / <alpha-value>)',
          700: 'rgb(var(--color-warning-rgb) / <alpha-value>)',
          800: 'rgb(var(--color-warning-rgb) / <alpha-value>)',
        },
      },
      boxShadow: {
        sm: 'var(--shadow-subtle)',
        DEFAULT: 'var(--shadow-subtle)',
        md: 'var(--shadow-subtle)',
        lg: 'var(--shadow-panel)',
        xl: 'var(--shadow-panel)',
      },
      borderRadius: { lg: '0.75rem', xl: '1rem', '2xl': '1.25rem' },
    },
  },
  plugins: [],
};
