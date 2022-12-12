import { defineConfig } from 'vite'
import path from 'path'

export default defineConfig({
  build:{
    lib:{
      entry:'./src/Lib.ts',
      name:'gameai',
      fileName:'gameai'
    },
    rollupOptions:{
    }
  },
  plugins:[
  ]
})
