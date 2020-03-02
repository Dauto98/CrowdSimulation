#include <Windows.h>
#include "MainWindow.h"

#include <QGraphicsView>
#include <QtGui>
#include <QBrush>

#include <iostream>

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

MainWindow::MainWindow(const Ped::Model &pedModel) : model(pedModel), agentsX(pedModel.getAgentsX()), agentsY(pedModel.getAgentsY())
{
	// The Window 
	graphicsView = new QGraphicsView();

	setCentralWidget(graphicsView);

	// A surface for managing a large number of 2D graphical items
	scene = new QGraphicsScene(QRect(0, 0, 800, 600), this);

	// Connect
	graphicsView->setScene(scene);

	// Paint on surface
	scene->setBackgroundBrush(Qt::white);

	for (int x = 0; x <= 800; x += cellsizePixel)
	{
		scene->addLine(x, 0, x, 600, QPen(Qt::gray));
	}

	// Now add the horizontal lines, paint them gray
	for (int y = 0; y <= 600; y += cellsizePixel)
	{
		scene->addLine(0, y, 800, y, QPen(Qt::gray));
	}

	QBrush greenBrush(Qt::green);
	QPen outlinePen(Qt::black);
	outlinePen.setWidth(2);
	
	for (int i = 0; i < agentsX.size(); i++) {
		rectList.push_back((scene->addRect(MainWindow::cellToPixel(agentsX[i]), MainWindow::cellToPixel(agentsY[i]), MainWindow::cellsizePixel - 1, MainWindow::cellsizePixel - 1, outlinePen, greenBrush)));
	}

	const int heatmapSize = model.getHeatmapSize();
	QPixmap pixmapDummy = QPixmap(heatmapSize, heatmapSize);
	pixmap = scene->addPixmap(pixmapDummy);

	paint();
	graphicsView->show(); // Redundant? 
}

void MainWindow::paint() {

	// Uncomment this to paint the heatmap (Assignment 4)
	const int heatmapSize = model.getHeatmapSize();
	QImage image((uchar*)model.cudaGetHeatmap(), heatmapSize, heatmapSize, heatmapSize * sizeof(int), QImage::Format_ARGB32);
	pixmap->setPixmap(QPixmap::fromImage(image));

	// Paint all agents: green, if the only agent on that position, otherwise red
	std::set<std::tuple<float, float>> positionsTaken;
	for (int i = 0; i < agentsX.size(); i++) {
		size_t tupleSizeBeforeInsert = positionsTaken.size();
		positionsTaken.insert(std::make_pair(agentsX[i], agentsY[i]));
		size_t tupleSizeAfterInsert = positionsTaken.size();

		QColor color;
		if (tupleSizeBeforeInsert != tupleSizeAfterInsert) {
			color = Qt::green;
		}
		else {
			color = Qt::red;
		}

		QBrush brush(color);
		rectList[i]->setBrush(brush);
		rectList[i]->setRect(MainWindow::cellToPixel(agentsX[i]), MainWindow::cellToPixel(agentsY[i]), MainWindow::cellsizePixel - 1, MainWindow::cellsizePixel - 1);
	}
}

int MainWindow::cellToPixel(int val)
{
	return val*cellsizePixel;
}
MainWindow::~MainWindow()
{
	for_each(viewAgents.begin(), viewAgents.end(), [](ViewAgent * agent){delete agent; });
}