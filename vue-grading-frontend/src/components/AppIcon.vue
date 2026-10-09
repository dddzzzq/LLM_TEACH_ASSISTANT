<template>
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path v-for="(path, index) in (paths[name] || paths.book)" :key="index" :d="path" />
  </svg>
</template>

<script setup>
defineProps({ name: { type: String, default: 'book' } })
const paths = {
  home: ['m3 10 9-7 9 7', 'M5 9v11h5v-6h4v6h5V9'],
  search: ['M10.5 18a7.5 7.5 0 1 0 0-15 7.5 7.5 0 0 0 0 15Z', 'm16 16 5 5'],
  check: ['M9 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-4', 'm9 12 3 3L22 5'],
  pen: ['m16 3 5 5-12 12-6 1 1-6L16 3Z', 'm13 6 5 5'],
  chat: ['M21 11.5a8.5 8.5 0 0 1-8.5 8.5H4l-3 2V11.5a8.5 8.5 0 0 1 17-1', 'M20 2v6', 'M17 5h6', 'M6 11h6', 'M6 15h9'],
  settings: ['M4 7h16', 'M4 17h16', 'M8 4v6', 'M16 14v6'],
  book: ['M12 5v16', 'M12 5C9 3 5 3 2 4v15c3-1 7-1 10 2 3-3 7-3 10-2V4c-3-1-7-1-10 1Z'],
  file: ['M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z', 'M14 2v6h6', 'M8 13h8', 'M8 17h5'],
  arrow: ['M4 12h16', 'm14 6 6 6-6 6'],
  sparkles: ['m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z', 'M21 2v4', 'M19 4h4'],
  shield: ['m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z', 'm8 12 3 3 5-6'],
  logout: ['M9 4H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h4', 'M9 12h12', 'm17 8 4 4-4 4'],
  download: ['M12 3v12', 'm7 10 5 5 5-5', 'M4 16v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4'],
  chart: ['M4 3v17h17', 'M9 15v-4', 'M14 15V7', 'M19 15V4'],
  plus: ['M12 5v14', 'M5 12h14'],
  history: ['M3 11a9 9 0 1 1 2.5 7', 'M3 4v7h7', 'M12 7v5l3 2'],
}
</script>
