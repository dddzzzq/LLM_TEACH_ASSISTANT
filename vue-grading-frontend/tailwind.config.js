/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        gray: {
          50: '#f7f9fb', 100: '#eff3f7', 200: '#e0e7ee', 300: '#c8d3de',
          400: '#8b9bac', 500: '#66788a', 600: '#4c6072', 700: '#344b5f',
          800: '#20384c', 900: '#142d40',
        },
        indigo: {
          50: '#edf7f5', 100: '#d9eeea', 200: '#b4ddd5', 300: '#83c4b9',
          400: '#4ba799', 500: '#278e82', 600: '#197b70', 700: '#14655d',
          800: '#135149', 900: '#133f3a',
        },
      },
      boxShadow: {
        sm: '0 1px 2px rgb(20 45 64 / 0.03)',
        DEFAULT: '0 2px 8px rgb(20 45 64 / 0.04)',
        md: '0 3px 14px rgb(20 45 64 / 0.045)',
        lg: '0 6px 20px rgb(20 45 64 / 0.055)',
        xl: '0 8px 28px rgb(20 45 64 / 0.06)',
      },
      borderRadius: { lg: '0.75rem', xl: '1rem', '2xl': '1.25rem' },
    },
  },
  plugins: [],
}
