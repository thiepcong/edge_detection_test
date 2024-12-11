package com.sample.edgedetection.processor

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt

const val TAG: String = "PaperProcessor"

fun processPicture(image: Bitmap): Corners? {
    // convert bitmap to OpenCV matrix
    val mat = Mat()
    Utils.bitmapToMat(image, mat)

    // shrink photo to make it easier to find document corners
    val shrunkImageHeight = 500.0
    Imgproc.resize(
        mat,
        mat,
        Size(
            shrunkImageHeight * image.width / image.height,
            shrunkImageHeight
        )
    )

    // convert photo to LUV colorspace to avoid glares caused by lights
    Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2Luv)

    // separate photo into 3 parts, (L, U, and V)
    val imageSplitByColorChannel: List<Mat> = mutableListOf()
    Core.split(mat, imageSplitByColorChannel)

    // find corners for each color channel, then pick the quad with the largest
    // area, and scale point to account for shrinking image before document detection
    val documentCorners: List<Point>? = imageSplitByColorChannel
        .mapNotNull { findCorners(it) }
        .maxByOrNull { Imgproc.contourArea(it) }
        ?.toList()
        ?.map {
            Point(
                it.x * image.height / shrunkImageHeight,
                it.y * image.height / shrunkImageHeight
            )
        }

    // sort points to force this order (top left, top right, bottom left, bottom right)
    val contours =  documentCorners
        ?.sortedBy { it.y }
        ?.chunked(2)
        ?.map { it.sortedBy { point -> point.x } }
        ?.flatten() ?: return null
    return Corners(contours, mat.size())
}

/**
 * take an image matrix with a document, and find the document's corners
 *
 * @param image a photo with a document in matrix format (only 1 color space)
 * @return a matrix with document corners or null if we can't find corners
 */
private fun findCorners(image: Mat): MatOfPoint? {
    val outputImage = Mat()

    // blur image to help remove noise
    Imgproc.GaussianBlur(image, outputImage, Size(5.0, 5.0),0.0)

    // convert all pixels to either black or white (document should be black after this), but
    // there might be other parts of the photo that turn black
    Imgproc.threshold(
        outputImage,
        outputImage,
        0.0,
        255.0,
        Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU
    )

    // detect the document's border using the Canny edge detection algorithm
    Imgproc.Canny(outputImage, outputImage, 50.0, 200.0)

    // the detect edges might have gaps, so try to close those
    Imgproc.morphologyEx(
        outputImage,
        outputImage,
        Imgproc.MORPH_CLOSE,
        Mat.ones(Size(5.0, 5.0), CvType.CV_8U)
    )

    // get outline of document edges, and outlines of other shapes in photo
    val contours: MutableList<MatOfPoint> = mutableListOf()
    Imgproc.findContours(
        outputImage,
        contours,
        Mat(),
        Imgproc.RETR_LIST,
        Imgproc.CHAIN_APPROX_SIMPLE
    )

    // approximate outlines using polygons
    var approxContours = contours.map {
        val approxContour = MatOfPoint2f()
        val contour2f = MatOfPoint2f(*it.toArray())
        Imgproc.approxPolyDP(
            contour2f,
            approxContour,
            0.02 * Imgproc.arcLength(contour2f, true),
            true
        )
        MatOfPoint(*approxContour.toArray())
    }

    // We now have many polygons, so remove polygons that don't have 4 sides since we
    // know the document has 4 sides. Calculate areas for all remaining polygons, and
    // remove polygons with small areas. We assume that the document takes up a large portion
    // of the photo. Remove polygons that aren't convex since a document can't be convex.
    approxContours = approxContours.filter {
        it.height() == 4 && Imgproc.contourArea(it) > 1000 && Imgproc.isContourConvex(it)
    }

    // Once we have all large, convex, 4-sided polygons find and return the 1 with the
    // largest area
    return approxContours.maxByOrNull { Imgproc.contourArea(it) }
}

fun cropPicture(picture: Mat, pts: List<Point>): Mat {

    pts.forEach { Log.i(TAG, "point: $it") }
    val tl = pts[0]
    val tr = pts[1]
    val br = pts[2]
    val bl = pts[3]

    val widthA = sqrt((br.x - bl.x).pow(2.0) + (br.y - bl.y).pow(2.0))
    val widthB = sqrt((tr.x - tl.x).pow(2.0) + (tr.y - tl.y).pow(2.0))

    val dw = max(widthA, widthB)
    val maxWidth = java.lang.Double.valueOf(dw).toInt()


    val heightA = sqrt((tr.x - br.x).pow(2.0) + (tr.y - br.y).pow(2.0))
    val heightB = sqrt((tl.x - bl.x).pow(2.0) + (tl.y - bl.y).pow(2.0))

    val dh = max(heightA, heightB)
    val maxHeight = java.lang.Double.valueOf(dh).toInt()

    val croppedPic = Mat(maxHeight, maxWidth, CvType.CV_8UC4)

    val srcMat = Mat(4, 1, CvType.CV_32FC2)
    val dstMat = Mat(4, 1, CvType.CV_32FC2)

    srcMat.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y)
    dstMat.put(0, 0, 0.0, 0.0, dw, 0.0, dw, dh, 0.0, dh)

    val m = Imgproc.getPerspectiveTransform(srcMat, dstMat)

    Imgproc.warpPerspective(picture, croppedPic, m, croppedPic.size())
    m.release()
    srcMat.release()
    dstMat.release()
    Log.i(TAG, "crop finish")
    return croppedPic
}

fun enhancePicture(src: Bitmap?): Bitmap {
    val srcMat = Mat()
    Utils.bitmapToMat(src, srcMat)
    Imgproc.cvtColor(srcMat, srcMat, Imgproc.COLOR_RGBA2GRAY)
    Imgproc.adaptiveThreshold(
        srcMat,
        srcMat,
        255.0,
        Imgproc.ADAPTIVE_THRESH_MEAN_C,
        Imgproc.THRESH_BINARY,
        15,
        15.0
    )
    val result = Bitmap.createBitmap(src?.width ?: 1080, src?.height ?: 1920, Bitmap.Config.RGB_565)
    Utils.matToBitmap(srcMat, result, true)
    srcMat.release()
    return result
}

private fun findContours(src: Mat): List<MatOfPoint> {

    val grayImage: Mat
    val cannedImage: Mat
    val kernel: Mat = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0))
    val dilate: Mat
    val size = Size(src.size().width, src.size().height)
    grayImage = Mat(size, CvType.CV_8UC4)
    cannedImage = Mat(size, CvType.CV_8UC1)
    dilate = Mat(size, CvType.CV_8UC1)

    Imgproc.cvtColor(src, grayImage, Imgproc.COLOR_BGR2GRAY)
    Imgproc.GaussianBlur(grayImage, grayImage, Size(5.0, 5.0), 0.0)
    Imgproc.threshold(grayImage, grayImage, 20.0, 255.0, Imgproc.THRESH_TRIANGLE)
    Imgproc.Canny(grayImage, cannedImage, 75.0, 200.0)
    Imgproc.dilate(cannedImage, dilate, kernel)
    val contours = ArrayList<MatOfPoint>()
    val hierarchy = Mat()
    Imgproc.findContours(
        dilate,
        contours,
        hierarchy,
        Imgproc.RETR_TREE,
        Imgproc.CHAIN_APPROX_SIMPLE
    )

    val filteredContours = contours
        .filter { p: MatOfPoint -> Imgproc.contourArea(p) > 100e2 }
        .sortedByDescending { p: MatOfPoint -> Imgproc.contourArea(p) }
        .take(25)

    hierarchy.release()
    grayImage.release()
    cannedImage.release()
    kernel.release()
    dilate.release()

    return filteredContours
}

private fun getCorners(contours: List<MatOfPoint>, size: Size): Corners? {
    val indexTo: Int = when (contours.size) {
        in 0..5 -> contours.size - 1
        else -> 4
    }
    for (index in 0..contours.size) {
        if (index in 0..indexTo) {
            val c2f = MatOfPoint2f(*contours[index].toArray())
            val peri = Imgproc.arcLength(c2f, true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(c2f, approx, 0.03 * peri, true)
            val points = approx.toArray().asList()
            val convex = MatOfPoint()
            approx.convertTo(convex, CvType.CV_32S)
            // select biggest 4 angles polygon
            if (points.size == 4 && Imgproc.isContourConvex(convex)) {
                val foundPoints = sortPoints(points)
                return Corners(foundPoints, size)
            }
        } else {
            return null
        }
    }

    return null
}
private fun sortPoints(points: List<Point>): List<Point> {
    val p0 = points.minByOrNull { point -> point.x + point.y } ?: Point()
    val p1 = points.minByOrNull { point: Point -> point.y - point.x } ?: Point()
    val p2 = points.maxByOrNull { point: Point -> point.x + point.y } ?: Point()
    val p3 = points.maxByOrNull { point: Point -> point.y - point.x } ?: Point()
    return listOf(p0, p1, p2, p3)
}
