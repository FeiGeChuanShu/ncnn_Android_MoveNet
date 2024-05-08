// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
package com.tencent.ncnnbodypose

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.PixelFormat
import android.os.Bundle
import android.support.v4.app.ActivityCompat
import android.support.v4.content.ContextCompat
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.AdapterView.OnItemSelectedListener
import android.widget.Button
import android.widget.Spinner

class MainActivity : Activity(), SurfaceHolder.Callback {
    private val ncnnbodypose = NcnnBodypose()
    private var facing = 0

    private var spinnerModel: Spinner? = null
    private var spinnerCPUGPU: Spinner? = null
    private var current_model = 0
    private var current_cpugpu = 0

    private var cameraView: SurfaceView? = null

    /** Called when the activity is first created.  */
    public override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        cameraView = findViewById<View>(R.id.cameraview) as SurfaceView

        cameraView!!.holder.setFormat(PixelFormat.RGBA_8888)
        cameraView!!.holder.addCallback(this)

        val buttonSwitchCamera = findViewById<View>(R.id.buttonSwitchCamera) as Button
        buttonSwitchCamera.setOnClickListener {
            val new_facing = 1 - facing
            ncnnbodypose.closeCamera()

            ncnnbodypose.openCamera(new_facing)
            facing = new_facing
        }

        spinnerModel = findViewById<View>(R.id.spinnerModel) as Spinner
        spinnerModel!!.onItemSelectedListener = object : OnItemSelectedListener {
            override fun onItemSelected(
                arg0: AdapterView<*>?, arg1: View, position: Int, id: Long
            ) {
                if (position != current_model) {
                    current_model = position
                    reload()
                }
            }

            override fun onNothingSelected(arg0: AdapterView<*>?) {
            }
        }

        spinnerCPUGPU = findViewById<View>(R.id.spinnerCPUGPU) as Spinner
        spinnerCPUGPU!!.onItemSelectedListener = object : OnItemSelectedListener {
            override fun onItemSelected(
                arg0: AdapterView<*>?, arg1: View, position: Int, id: Long
            ) {
                if (position != current_cpugpu) {
                    current_cpugpu = position
                    reload()
                }
            }

            override fun onNothingSelected(arg0: AdapterView<*>?) {
            }
        }

        reload()
    }

    private fun reload() {
        val ret_init = ncnnbodypose.loadModel(assets, current_model, current_cpugpu)
        if (!ret_init) {
            Log.e("MainActivity", "ncnnbodypose loadModel failed")
        }
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        ncnnbodypose.setOutputWindow(holder.surface)
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
    }

    public override fun onResume() {
        super.onResume()

        if (ContextCompat.checkSelfPermission(
                applicationContext, Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_DENIED
        ) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA
            )
        }

        ncnnbodypose.openCamera(facing)
    }

    public override fun onPause() {
        super.onPause()

        ncnnbodypose.closeCamera()
    }

    companion object {
        const val REQUEST_CAMERA: Int = 100
    }
}
