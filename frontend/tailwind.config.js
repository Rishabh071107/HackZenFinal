/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        pitch: {
          950: "#050b06",
          900: "#0b1f10",
          800: "#123016",
          700: "#1a4520",
        },
        accent: {
          500: "#22c55e",
          400: "#4ade80",
        },
      },
    },
  },
  plugins: [],
};
