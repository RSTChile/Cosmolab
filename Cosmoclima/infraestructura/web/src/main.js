import { mount } from 'svelte';
import './estilos.css';
import App from './App.svelte';

export default mount(App, { target: document.getElementById('app') });
