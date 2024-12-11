package com.sample.edgedetection.utils

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import java.io.File
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

/**
 * This class contains helper functions for processing images
 *
 * @constructor creates image util
 */
class ImageUtil {
    /**
     * get image matrix from file path
     *
     * @param filePath image is saved here
     * @return image matrix
     */
    private fun getImageMatrixFromFilePath(filePath: String): Mat {
        // read image as matrix using OpenCV
        val image: Mat = Imgcodecs.imread(filePath)

        // if OpenCV fails to read the image then it's empty
        if (!image.empty()) {
            // convert image to RGB color space since OpenCV reads it using BGR color space
            Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB)
            return image
        }

        if (!File(filePath).exists()) {
            throw Exception("File doesn't exist - $filePath")
        }

        if (!File(filePath).canRead()) {
            throw Exception("You don't have permission to read $filePath")
        }

        // try reading image without OpenCV
        var imageBitmap = BitmapFactory.decodeFile(filePath)
        val rotation = when (ExifInterface(filePath).getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90
            ExifInterface.ORIENTATION_ROTATE_180 -> 180
            ExifInterface.ORIENTATION_ROTATE_270 -> 270
            else -> 0
        }
        imageBitmap = Bitmap.createBitmap(
            imageBitmap,
            0,
            0,
            imageBitmap.width,
            imageBitmap.height,
            Matrix().apply { postRotate(rotation.toFloat()) },
            true
        )
        Utils.bitmapToMat(imageBitmap, image)

        return image
    }

    /**
     * get bitmap image from file path
     *
     * @param filePath image is saved here
     * @return image bitmap
     */
    fun getImageFromFilePath(filePath: String): Bitmap {
        // read image as matrix using OpenCV
        val image: Mat = this.getImageMatrixFromFilePath(filePath)

        // convert image matrix to bitmap
        val bitmap = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(image, bitmap)
        return bitmap
    }

    /**
     * take a photo with a document, crop everything out but document, and force it to display
     * as a rectangle
     *
     * @param photoFilePath original image is saved here
     * @param corners the 4 document corners
     * @return bitmap with cropped and warped document
     */


    /**
     * get bitmap image from file uri
     *
     * @param fileUriString image is saved here and starts with file:///
     * @return bitmap image
     */
    fun readBitmapFromFileUriString(
        fileUriString: String,
        contentResolver: ContentResolver
    ): Bitmap {
        return BitmapFactory.decodeStream(
            contentResolver.openInputStream(Uri.parse(fileUriString))
        )
    }
}