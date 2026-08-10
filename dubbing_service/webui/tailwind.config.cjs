/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./app.js"],
  theme: {
    extend: {
      colors: {
        primary: {
          50: "#ecfdf5",
          100: "#d1fae5",
          200: "#a7f3d0",
          300: "#6ee7b7",
          400: "#34d399",
          500: "#22c28f",
          600: "#16a176",
          700: "#15805f",
          800: "#166152",
          900: "#145044"
        }
      }
    }
  },
  plugins: []
};
