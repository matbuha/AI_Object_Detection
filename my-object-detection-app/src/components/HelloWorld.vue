<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
    <h3>testing flask and camera</h3>
    <ul></ul>
    <button @click="startCamera">Start Camera</button>
    <ul></ul>
    <button @click="stopCamera">Stop Camera</button>
  </div>
</template>

<script>
import axios from 'axios'; // Import Axios

export default {
  name: 'HelloWorld',
  methods: {
    startCamera() {
      axios.get('http://localhost:5000/start')
        .then(response => {
          console.log('Camera started:', response);
        })
        .catch(error => {
          console.error('Error starting camera:', error);
        });
    },
    stopCamera() {
      axios.get('http://localhost:5000/stop')
        .then(response => {
          console.log('Camera stopped:', response);
        })
        .catch(error => {
          console.error('Error stopping camera:', error);
        });
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
